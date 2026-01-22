#!/usr/bin/env python3
"""
Validation Visualizer implementation for the Simulation Accuracy and Validation Framework.

This module provides a comprehensive visualization system for rendering various types of
visualizations for simulation validation results, including comparison charts, distribution
plots, time-series visualizations, and interactive dashboards.
"""

import os
import logging
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, cast
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("validation_visualizer")

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

# Optional visualization dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    matplotlib_available = True
except ImportError:
    logger.warning("Matplotlib not available. Basic plotting will not work.")
    matplotlib_available = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    from plotly.subplots import make_subplots
    plotly_available = True
except ImportError:
    logger.warning("Plotly not available. Interactive plots will not work.")
    plotly_available = False

try:
    import pandas as pd
    pandas_available = True
except ImportError:
    logger.warning("Pandas not available. Data manipulation for visualization will be limited.")
    pandas_available = False

try:
    import numpy as np
    numpy_available = True
except ImportError:
    logger.warning("NumPy not available. Statistical visualizations will be limited.")
    numpy_available = False

try:
    from scipy import stats
    scipy_available = True
except ImportError:
    logger.warning("SciPy not available. Advanced statistical visualizations will be limited.")
    scipy_available = False


class ValidationVisualizer:
    """
    Generates visualizations for simulation validation results.
    
    This class provides methods for creating various types of visualizations for
    simulation validation results, including comparison charts, distribution plots,
    time-series visualizations, and interactive dashboards.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation visualizer.
        
        Args:
            config: Configuration options for the visualizer
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "output_directory": "output/visualizations",
            "default_width": 1000,
            "default_height": 600,
            "default_format": "html",
            "export_formats": ["html", "png", "pdf", "svg"],  # Added support for multiple export formats
            "color_scheme": "viridis",
            "theme": "plotly_white",
            "use_interactive": True,
            "show_grid": True,
            "font_size": 12,
            "title_font_size": 16,
            "dpi": 150,
            "transparent_bg": False,
            "include_timestamp": True,
            "include_export_menu": True,
            "max_points_scatter": 1000,
            "max_categories_bar": 15,
            "include_watermark": False,
            "watermark_text": "Simulation Validation Framework",
            "display_powered_by": True,
            "animated_transitions": True,  # Added support for animated transitions
            "animation_duration": 500,     # Duration of animations in milliseconds
            "animation_easing": "cubic-in-out"  # Animation easing function
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Set up plotly theme if available
        if plotly_available:
            theme = self.config.get("theme", "plotly_white")
            try:
                pio.templates.default = theme
            except Exception as e:
                logger.warning(f"Could not set Plotly theme: {e}")
    
    def export_visualization(
        self, 
        fig: Any, 
        output_path: Optional[str] = None, 
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export a visualization to multiple formats.
        
        Args:
            fig: The figure object (either Plotly or Matplotlib)
            output_path: Base path for exporting the visualization
            formats: List of formats to export (e.g., ["html", "png", "pdf", "svg"])
            
        Returns:
            Dictionary mapping formats to file paths
        """
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config["output_directory"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"visualization_{timestamp}")
        
        # Default to all configured formats if none specified
        if formats is None:
            formats = self.config["export_formats"]
            
        exported_files = {}
        
        # Check if we're dealing with a Plotly figure
        is_plotly = plotly_available and hasattr(fig, "write_html")
        
        for fmt in formats:
            try:
                if is_plotly:
                    # Export using Plotly
                    if fmt.lower() == "html":
                        file_path = f"{output_path}.html"
                        fig.write_html(
                            file_path, 
                            include_plotlyjs=True,
                            full_html=True,
                            include_mathjax=False,
                            config={
                                "displayModeBar": True,
                                "responsive": True,
                                "displaylogo": False,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": Path(output_path).name,
                                    "height": self.config["default_height"],
                                    "width": self.config["default_width"],
                                    "scale": 2
                                }
                            }
                        )
                        exported_files["html"] = file_path
                    elif fmt.lower() in ["png", "jpg", "jpeg", "webp", "svg", "pdf"]:
                        file_path = f"{output_path}.{fmt}"
                        fig.write_image(
                            file_path,
                            format=fmt,
                            width=self.config["default_width"],
                            height=self.config["default_height"],
                            scale=2
                        )
                        exported_files[fmt] = file_path
                    else:
                        logger.warning(f"Unsupported export format for Plotly: {fmt}")
                else:
                    # Export using Matplotlib
                    if matplotlib_available and hasattr(fig, "savefig"):
                        file_path = f"{output_path}.{fmt}"
                        fig.savefig(
                            file_path,
                            format=fmt,
                            dpi=self.config["dpi"],
                            bbox_inches="tight",
                            transparent=self.config["transparent_bg"]
                        )
                        exported_files[fmt] = file_path
                    else:
                        logger.warning(f"Cannot export to {fmt}: matplotlib not available or invalid figure object")
            except Exception as e:
                logger.error(f"Error exporting to {fmt}: {str(e)}")
                
        return exported_files
        
    def create_animated_time_series(
        self,
        validation_results: List[ValidationResult],
        metric_name: str,
        hardware_id: str,
        model_id: str,
        show_trend: bool = True,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        formats: Optional[List[str]] = None,
        frame_duration: int = 100,
        transition_duration: int = 300
    ) -> Union[Dict[str, str], None]:
        """
        Create an animated time series visualization showing the evolution of simulation accuracy over time.
        
        Args:
            validation_results: List of validation results
            metric_name: Name of the metric to visualize
            hardware_id: Hardware ID to filter results by
            model_id: Model ID to filter results by
            show_trend: Whether to show a trend line
            output_path: Path to save the chart
            title: Custom title for the chart
            formats: List of formats to export the visualization to
            frame_duration: Duration of each frame in milliseconds
            transition_duration: Duration of transitions between frames in milliseconds
            
        Returns:
            Dictionary of exported files or None if unsuccessful
        """
        if not validation_results:
            logger.warning("No validation results provided for animated time series")
            return None
        
        # Check if plotly is available (required for animations)
        if not plotly_available:
            logger.warning("Plotly not available. Cannot create animated time series.")
            return None
            
        if not pandas_available:
            logger.warning("Pandas not available. Cannot create animated time series.")
            return None
        
        # Filter and sort validation results
        filtered_results = []
        for val_result in validation_results:
            if (val_result.hardware_result.hardware_id == hardware_id and
                val_result.simulation_result.model_id == model_id):
                filtered_results.append(val_result)
        
        if not filtered_results:
            logger.warning(f"No validation results found for hardware '{hardware_id}' and model '{model_id}'")
            return None
        
        # Sort by timestamp
        filtered_results.sort(key=lambda x: x.validation_timestamp)
        
        # Extract time series data
        data = []
        for val_result in filtered_results:
            if metric_name in val_result.metrics_comparison:
                comparison = val_result.metrics_comparison[metric_name]
                
                sim_value = comparison.get("simulation_value", 0)
                hw_value = comparison.get("hardware_value", 0)
                mape = comparison.get("mape", 0)
                
                data.append({
                    "timestamp": val_result.validation_timestamp,
                    "simulation_value": sim_value,
                    "hardware_value": hw_value,
                    "mape": mape,
                    "error": hw_value - sim_value if hw_value is not None and sim_value is not None else None
                })
        
        if not data:
            logger.warning(f"No time series data found for metric '{metric_name}'")
            return None
        
        # Convert to dataframe
        df = pd.DataFrame(data)
        
        # Convert timestamps to datetime
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            logger.warning(f"Could not convert timestamps: {e}")
        
        # Generate chart title
        if title is None:
            title = f"Animated Time Series: {metric_name} for {model_id} on {hardware_id}"
        
        # Create the animation
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create figure with two subplots: one for values, one for MAPE
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=["Simulation vs Hardware Values", "Mean Absolute Percentage Error (MAPE)"],
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Create frames for animation
            frames = []
            
            for i in range(1, len(df) + 1):
                subset = df.iloc[:i]
                
                # Create frame for this subset
                frame = {"data": [], "name": str(i)}
                
                # Add simulation values trace
                frame["data"].append(go.Scatter(
                    x=subset["timestamp"],
                    y=subset["simulation_value"],
                    mode="lines+markers",
                    name="Simulation",
                    line=dict(color="blue"),
                    showlegend=False
                ))
                
                # Add hardware values trace
                frame["data"].append(go.Scatter(
                    x=subset["timestamp"],
                    y=subset["hardware_value"],
                    mode="lines+markers",
                    name="Hardware",
                    line=dict(color="red"),
                    showlegend=False
                ))
                
                # Add MAPE trace
                frame["data"].append(go.Scatter(
                    x=subset["timestamp"],
                    y=subset["mape"],
                    mode="lines+markers",
                    name="MAPE",
                    line=dict(color="green"),
                    showlegend=False
                ))
                
                # Add trend line if requested
                if show_trend and len(subset) > 1:
                    import numpy as np
                    from scipy import stats
                    
                    # Add trend for MAPE
                    try:
                        x = np.array(range(len(subset)))
                        y = subset["mape"].values
                        slope, intercept, _, _, _ = stats.linregress(x, y)
                        trend_y = intercept + slope * x
                        
                        frame["data"].append(go.Scatter(
                            x=subset["timestamp"],
                            y=trend_y,
                            mode="lines",
                            name="MAPE Trend",
                            line=dict(color="darkgreen", dash="dash"),
                            showlegend=False
                        ))
                    except Exception as e:
                        logger.warning(f"Could not calculate trend line: {e}")
                
                frames.append(frame)
            
            # Add initial traces
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"].iloc[:1],
                    y=df["simulation_value"].iloc[:1],
                    mode="lines+markers",
                    name="Simulation",
                    line=dict(color="blue")
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"].iloc[:1],
                    y=df["hardware_value"].iloc[:1],
                    mode="lines+markers",
                    name="Hardware",
                    line=dict(color="red")
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"].iloc[:1],
                    y=df["mape"].iloc[:1],
                    mode="lines+markers",
                    name="MAPE",
                    line=dict(color="green")
                ),
                row=2, col=1
            )
            
            # Add trend line for initial frame if requested
            if show_trend and len(df) > 1:
                import numpy as np
                from scipy import stats
                
                # Add trend for MAPE
                try:
                    x = np.array([0])
                    y = df["mape"].iloc[:1].values
                    trend_y = y  # Just a point for the first frame
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"].iloc[:1],
                            y=trend_y,
                            mode="lines",
                            name="MAPE Trend",
                            line=dict(color="darkgreen", dash="dash")
                        ),
                        row=2, col=1
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate initial trend line: {e}")
            
            # Configure animation settings
            animation_settings = {
                "frame": {"duration": frame_duration, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": transition_duration, "easing": "cubic-in-out"}
            }
            
            # Add slider and play button for animation control
            sliders = [{
                "active": 0,
                "steps": [
                    {
                        "args": [
                            [frame["name"]],
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": transition_duration}
                            }
                        ],
                        "label": frame["name"],
                        "method": "animate"
                    }
                    for frame in frames
                ],
                "x": 0.1,
                "y": 0,
                "currentvalue": {
                    "font": {"size": 12},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "center"
                },
                "len": 0.9
            }]
            
            # Configure update menu for play/pause control
            updatemenus = [{
                "buttons": [
                    {
                        "args": [None, animation_settings],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }]
            
            # Configure the layout
            fig.update_layout(
                title=title,
                height=self.config["default_height"],
                width=self.config["default_width"],
                showlegend=True,
                updatemenus=updatemenus,
                sliders=sliders
            )
            
            # Set axes titles
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
            
            # Add frames to the figure
            fig.frames = frames
            
            # Export the animation
            return self.export_visualization(fig, output_path, formats)
            
        except Exception as e:
            logger.error(f"Error creating animated time series: {str(e)}")
            return None
            
    def create_3d_error_visualization(
        self,
        validation_results: List[ValidationResult],
        metrics: List[str],
        hardware_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        formats: Optional[List[str]] = None
    ) -> Union[Dict[str, str], None]:
        """
        Create a 3D visualization comparing errors across multiple metrics.
        This allows for a more comprehensive view of simulation accuracy across dimensions.
        
        Args:
            validation_results: List of validation results
            metrics: List of metrics to include in the visualization (max 3)
            hardware_ids: List of hardware IDs to filter by
            model_ids: List of model IDs to filter by
            interactive: Whether to create an interactive plot
            output_path: Path to save the chart
            title: Custom title for the chart
            formats: List of formats to export the visualization to
            
        Returns:
            Dictionary of exported files or None if unsuccessful
        """
        if not validation_results:
            logger.warning("No validation results provided for 3D error visualization")
            return None
        
        # Check if we have the required number of metrics
        if len(metrics) < 2 or len(metrics) > 3:
            logger.warning("3D error visualization requires 2 or 3 metrics")
            return None
        
        # Check if plotly is available (required for 3D plots)
        if not plotly_available:
            logger.warning("Plotly not available. Cannot create 3D error visualization.")
            return None
            
        if not pandas_available:
            logger.warning("Pandas not available. Cannot create 3D error visualization.")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Force interactive for 3D plots
        if not use_interactive:
            logger.warning("3D visualization requires interactive mode. Enabling interactive mode.")
            use_interactive = True
        
        # Filter validation results by hardware_ids and model_ids if provided
        filtered_results = validation_results
        
        if hardware_ids:
            filtered_results = [r for r in filtered_results 
                              if r.hardware_result.hardware_id in hardware_ids]
        
        if model_ids:
            filtered_results = [r for r in filtered_results 
                              if r.simulation_result.model_id in model_ids]
        
        if not filtered_results:
            logger.warning("No validation results found after filtering")
            return None
        
        # Extract data for visualization
        data = []
        for val_result in filtered_results:
            row = {
                "hardware_id": val_result.hardware_result.hardware_id,
                "model_id": val_result.simulation_result.model_id,
                "batch_size": val_result.hardware_result.batch_size,
                "precision": val_result.hardware_result.precision,
                "timestamp": val_result.validation_timestamp
            }
            
            # Extract metrics data
            metrics_available = True
            for metric in metrics:
                if metric in val_result.metrics_comparison:
                    comparison = val_result.metrics_comparison[metric]
                    row[f"{metric}_mape"] = comparison.get("mape", 0)
                    row[f"{metric}_sim"] = comparison.get("simulation_value", 0)
                    row[f"{metric}_hw"] = comparison.get("hardware_value", 0)
                else:
                    metrics_available = False
                    break
            
            if metrics_available:
                data.append(row)
        
        if not data:
            logger.warning("No data available for the specified metrics")
            return None
        
        # Convert to dataframe
        df = pd.DataFrame(data)
        
        # Generate chart title
        if title is None:
            if len(metrics) == 2:
                title = f"2D MAPE Comparison: {metrics[0]} vs {metrics[1]}"
            else:
                title = f"3D MAPE Comparison: {metrics[0]} vs {metrics[1]} vs {metrics[2]}"
        
        # Create the visualization
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Create figure
            if len(metrics) == 3:
                # 3D scatter plot for 3 metrics
                fig = go.Figure(data=[go.Scatter3d(
                    x=df[f"{metrics[0]}_mape"],
                    y=df[f"{metrics[1]}_mape"],
                    z=df[f"{metrics[2]}_mape"],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df[f"{metrics[0]}_mape"],  # color by first metric
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title="MAPE (%)"),
                        showscale=True
                    ),
                    text=[f"Hardware: {h}<br>Model: {m}<br>Batch: {b}<br>Precision: {p}<br>"
                         f"{metrics[0]} MAPE: {df.iloc[i][f'{metrics[0]}_mape']:.2f}%<br>"
                         f"{metrics[1]} MAPE: {df.iloc[i][f'{metrics[1]}_mape']:.2f}%<br>"
                         f"{metrics[2]} MAPE: {df.iloc[i][f'{metrics[2]}_mape']:.2f}%"
                         for i, (h, m, b, p) in enumerate(zip(df['hardware_id'], df['model_id'], 
                                                           df['batch_size'], df['precision']))],
                    hoverinfo='text'
                )])
                
                # Update layout for 3D plot
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=f"{metrics[0]} MAPE (%)",
                        yaxis_title=f"{metrics[1]} MAPE (%)",
                        zaxis_title=f"{metrics[2]} MAPE (%)",
                        xaxis=dict(autorange=True),
                        yaxis=dict(autorange=True),
                        zaxis=dict(autorange=True),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    height=800,
                    width=1000,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
            else:
                # 2D scatter plot with enhanced visuals for 2 metrics
                # Calculate the overall MAPE (average of both metrics)
                df['overall_mape'] = df[[f"{metrics[0]}_mape", f"{metrics[1]}_mape"]].mean(axis=1)
                
                # Create scatter plot
                fig = px.scatter(
                    df, 
                    x=f"{metrics[0]}_mape", 
                    y=f"{metrics[1]}_mape",
                    color='overall_mape',
                    size='overall_mape',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    hover_name='model_id',
                    hover_data=['hardware_id', 'batch_size', 'precision', 
                               f"{metrics[0]}_mape", f"{metrics[1]}_mape"],
                    title=title
                )
                
                # Add a reference line for equal MAPE
                max_val = max(df[f"{metrics[0]}_mape"].max(), df[f"{metrics[1]}_mape"].max())
                min_val = min(df[f"{metrics[0]}_mape"].min(), df[f"{metrics[1]}_mape"].min())
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
                    name='Equal MAPE'
                ))
                
                # Add shapes for reference zones
                fig.add_shape(
                    type="rect",
                    x0=0, y0=0,
                    x1=5, y1=5,
                    fillcolor="rgba(0,255,0,0.1)",
                    line=dict(width=0),
                    layer="below"
                )
                
                fig.add_shape(
                    type="rect",
                    x0=5, y0=5,
                    x1=10, y1=10,
                    fillcolor="rgba(255,255,0,0.1)",
                    line=dict(width=0),
                    layer="below"
                )
                
                fig.add_shape(
                    type="rect",
                    x0=10, y0=10,
                    x1=max_val * 1.1, y1=max_val * 1.1,
                    fillcolor="rgba(255,0,0,0.1)",
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add annotations
                fig.add_annotation(
                    x=2.5, y=2.5,
                    text="Excellent",
                    showarrow=False,
                    font=dict(color="green")
                )
                
                fig.add_annotation(
                    x=7.5, y=7.5,
                    text="Good",
                    showarrow=False,
                    font=dict(color="orange")
                )
                
                fig.add_annotation(
                    x=15, y=15,
                    text="Poor",
                    showarrow=False,
                    font=dict(color="red")
                )
                
                # Update layout
                fig.update_layout(
                    height=700,
                    width=900,
                    xaxis_title=f"{metrics[0]} MAPE (%)",
                    yaxis_title=f"{metrics[1]} MAPE (%)",
                    coloraxis_colorbar=dict(title="Overall MAPE (%)"),
                    showlegend=True
                )
            
            # Export the visualization
            return self.export_visualization(fig, output_path, formats)
            
        except Exception as e:
            logger.error(f"Error creating 3D error visualization: {str(e)}")
            return None
    
    def create_mape_comparison_chart(
        self,
        validation_results: List[ValidationResult],
        metric_name: str = "all",
        hardware_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        sort_by: str = "hardware",
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart comparing MAPE values across different hardware/models.
        
        Args:
            validation_results: List of validation results
            metric_name: Name of the metric to visualize, or "all" for average
            hardware_ids: List of hardware IDs to include (if None, include all)
            model_ids: List of model IDs to include (if None, include all)
            sort_by: Sort results by "hardware", "model", or "value"
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for MAPE comparison chart")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Extract data
        data = []
        for val_result in validation_results:
            hw_id = val_result.hardware_result.hardware_id
            model_id = val_result.hardware_result.model_id
            
            # Skip if not in the specified hardware/model IDs
            if hardware_ids and hw_id not in hardware_ids:
                continue
            if model_ids and model_id not in model_ids:
                continue
            
            # Extract MAPE values
            if metric_name == "all":
                # Calculate average MAPE across all metrics
                mape_values = []
                for metric, comparison in val_result.metrics_comparison.items():
                    if "mape" in comparison:
                        mape_values.append(comparison["mape"])
                
                if mape_values:
                    avg_mape = sum(mape_values) / len(mape_values)
                    data.append({
                        "hardware_id": hw_id,
                        "model_id": model_id,
                        "metric": "average",
                        "mape": avg_mape
                    })
            else:
                # Extract MAPE for the specified metric
                if metric_name in val_result.metrics_comparison and "mape" in val_result.metrics_comparison[metric_name]:
                    mape = val_result.metrics_comparison[metric_name]["mape"]
                    data.append({
                        "hardware_id": hw_id,
                        "model_id": model_id,
                        "metric": metric_name,
                        "mape": mape
                    })
        
        if not data:
            logger.warning(f"No MAPE data found for metric '{metric_name}'")
            return None
        
        # Convert to pandas DataFrame if available
        if pandas_available:
            df = pd.DataFrame(data)
            
            # Sort data
            if sort_by == "hardware":
                df = df.sort_values("hardware_id")
            elif sort_by == "model":
                df = df.sort_values("model_id")
            elif sort_by == "value":
                df = df.sort_values("mape")
            
            # Generate chart title
            if title is None:
                if metric_name == "all":
                    title = "Average MAPE Across All Metrics"
                else:
                    title = f"MAPE for {metric_name}"
            
            # Create chart
            if use_interactive and plotly_available:
                return self._create_interactive_mape_chart(df, title, output_path)
            else:
                return self._create_static_mape_chart(df, title, output_path)
        else:
            # Basic implementation without pandas
            logger.warning("Pandas not available. Using basic implementation for MAPE chart.")
            
            # Sort data
            if sort_by == "hardware":
                data.sort(key=lambda x: x["hardware_id"])
            elif sort_by == "model":
                data.sort(key=lambda x: x["model_id"])
            elif sort_by == "value":
                data.sort(key=lambda x: x["mape"])
            
            # Generate chart title
            if title is None:
                if metric_name == "all":
                    title = "Average MAPE Across All Metrics"
                else:
                    title = f"MAPE for {metric_name}"
            
            # Create chart using basic matplotlib if available
            if matplotlib_available:
                return self._create_basic_mape_chart(data, title, output_path)
            else:
                logger.error("Cannot create MAPE chart: required dependencies not available")
                return None
    
    def create_metric_comparison_chart(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str,
        metrics: Optional[List[str]] = None,
        show_absolute_values: bool = False,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart comparing simulation vs hardware values for specific metrics.
        
        Args:
            validation_results: List of validation results
            hardware_id: Hardware ID to visualize
            model_id: Model ID to visualize
            metrics: List of metrics to include (if None, include common metrics)
            show_absolute_values: Whether to show absolute values or normalized
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for metric comparison chart")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Filter validation results by hardware and model
        filtered_results = []
        for val_result in validation_results:
            if (val_result.hardware_result.hardware_id == hardware_id and
                val_result.hardware_result.model_id == model_id):
                filtered_results.append(val_result)
        
        if not filtered_results:
            logger.warning(f"No validation results found for hardware '{hardware_id}' and model '{model_id}'")
            return None
        
        # Use the most recent validation result
        filtered_results.sort(key=lambda x: x.validation_timestamp, reverse=True)
        val_result = filtered_results[0]
        
        # Determine metrics to include
        common_metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", "power_consumption_w"]
        
        if metrics is None:
            # Use common metrics that are present in both simulation and hardware results
            metrics = []
            for metric in common_metrics:
                if (metric in val_result.simulation_result.metrics and
                    metric in val_result.hardware_result.metrics):
                    metrics.append(metric)
        
        if not metrics:
            logger.warning("No common metrics found for comparison")
            return None
        
        # Extract data
        data = []
        for metric in metrics:
            if (metric in val_result.simulation_result.metrics and
                metric in val_result.hardware_result.metrics):
                sim_value = val_result.simulation_result.metrics[metric]
                hw_value = val_result.hardware_result.metrics[metric]
                
                data.append({
                    "metric": metric,
                    "simulation": sim_value,
                    "hardware": hw_value
                })
        
        if not data:
            logger.warning("No matching metrics found for comparison")
            return None
        
        # Generate chart title
        if title is None:
            title = f"Simulation vs Hardware: {model_id} on {hardware_id}"
        
        # Create chart
        if pandas_available:
            df = pd.DataFrame(data)
            
            if use_interactive and plotly_available:
                return self._create_interactive_metric_comparison(df, title, show_absolute_values, output_path)
            else:
                return self._create_static_metric_comparison(df, title, show_absolute_values, output_path)
        else:
            # Basic implementation without pandas
            if matplotlib_available:
                return self._create_basic_metric_comparison(data, title, show_absolute_values, output_path)
            else:
                logger.error("Cannot create metric comparison chart: required dependencies not available")
                return None
    
    def create_error_distribution_chart(
        self,
        validation_results: List[ValidationResult],
        metric_name: str,
        hardware_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart showing the distribution of errors for a specific metric.
        
        Args:
            validation_results: List of validation results
            metric_name: Name of the metric to visualize
            hardware_ids: List of hardware IDs to include (if None, include all)
            model_ids: List of model IDs to include (if None, include all)
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for error distribution chart")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Extract error data
        data = []
        for val_result in validation_results:
            hw_id = val_result.hardware_result.hardware_id
            model_id = val_result.hardware_result.model_id
            
            # Skip if not in the specified hardware/model IDs
            if hardware_ids and hw_id not in hardware_ids:
                continue
            if model_ids and model_id not in model_ids:
                continue
            
            # Extract error values
            if metric_name in val_result.metrics_comparison:
                comparison = val_result.metrics_comparison[metric_name]
                
                if "error" in comparison:
                    data.append({
                        "hardware_id": hw_id,
                        "model_id": model_id,
                        "error": comparison["error"],
                        "mape": comparison.get("mape")
                    })
        
        if not data:
            logger.warning(f"No error data found for metric '{metric_name}'")
            return None
        
        # Generate chart title
        if title is None:
            title = f"Error Distribution for {metric_name}"
        
        # Create chart
        if pandas_available and numpy_available:
            df = pd.DataFrame(data)
            
            if use_interactive and plotly_available:
                return self._create_interactive_error_distribution(df, title, metric_name, output_path)
            else:
                return self._create_static_error_distribution(df, title, metric_name, output_path)
        else:
            # Basic implementation without pandas/numpy
            if matplotlib_available:
                return self._create_basic_error_distribution(data, title, metric_name, output_path)
            else:
                logger.error("Cannot create error distribution chart: required dependencies not available")
                return None
    
    def create_time_series_chart(
        self,
        validation_results: List[ValidationResult],
        metric_name: str,
        hardware_id: str,
        model_id: str,
        show_trend: bool = True,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a time series chart showing how simulation accuracy changes over time.
        
        Args:
            validation_results: List of validation results
            metric_name: Name of the metric to visualize
            hardware_id: Hardware ID to visualize
            model_id: Model ID to visualize
            show_trend: Whether to show a trend line
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for time series chart")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Filter and sort validation results
        filtered_results = []
        for val_result in validation_results:
            if (val_result.hardware_result.hardware_id == hardware_id and
                val_result.hardware_result.model_id == model_id):
                filtered_results.append(val_result)
        
        if not filtered_results:
            logger.warning(f"No validation results found for hardware '{hardware_id}' and model '{model_id}'")
            return None
        
        # Sort by timestamp
        filtered_results.sort(key=lambda x: x.validation_timestamp)
        
        # Extract time series data
        data = []
        for val_result in filtered_results:
            if metric_name in val_result.metrics_comparison:
                comparison = val_result.metrics_comparison[metric_name]
                
                if "mape" in comparison:
                    data.append({
                        "timestamp": val_result.validation_timestamp,
                        "mape": comparison["mape"],
                        "error": comparison.get("error")
                    })
        
        if not data:
            logger.warning(f"No time series data found for metric '{metric_name}'")
            return None
        
        # Generate chart title
        if title is None:
            title = f"Time Series: {metric_name} MAPE for {model_id} on {hardware_id}"
        
        # Create chart
        if pandas_available:
            # Convert timestamps to datetime
            df = pd.DataFrame(data)
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception as e:
                logger.warning(f"Could not convert timestamps: {e}")
            
            if use_interactive and plotly_available:
                return self._create_interactive_time_series(df, title, metric_name, show_trend, output_path)
            else:
                return self._create_static_time_series(df, title, metric_name, show_trend, output_path)
        else:
            # Basic implementation without pandas
            if matplotlib_available:
                return self._create_basic_time_series(data, title, metric_name, show_trend, output_path)
            else:
                logger.error("Cannot create time series chart: required dependencies not available")
                return None
    
    def create_hardware_comparison_heatmap(
        self,
        validation_results: List[ValidationResult],
        metric_name: str = "all",
        model_ids: Optional[List[str]] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a heatmap comparing simulation accuracy across hardware types.
        
        Args:
            validation_results: List of validation results
            metric_name: Name of the metric to visualize, or "all" for average
            model_ids: List of model IDs to include (if None, include all)
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for hardware comparison heatmap")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Extract unique hardware and model IDs
        hw_ids = set()
        mdl_ids = set()
        
        for val_result in validation_results:
            hw_ids.add(val_result.hardware_result.hardware_id)
            mdl_ids.add(val_result.hardware_result.model_id)
        
        # Filter model IDs if specified
        if model_ids:
            mdl_ids = set(model_id for model_id in mdl_ids if model_id in model_ids)
        
        if not hw_ids or not mdl_ids:
            logger.warning("No hardware or model IDs found for heatmap")
            return None
        
        # Create a matrix of MAPE values
        hw_list = sorted(list(hw_ids))
        mdl_list = sorted(list(mdl_ids))
        
        # Initialize matrix with NaN
        if numpy_available:
            matrix = np.full((len(hw_list), len(mdl_list)), np.nan)
        else:
            matrix = [[None for _ in range(len(mdl_list))] for _ in range(len(hw_list))]
        
        # Populate matrix with MAPE values
        for i, hw_id in enumerate(hw_list):
            for j, mdl_id in enumerate(mdl_list):
                # Find relevant validation results
                relevant_results = []
                for val_result in validation_results:
                    if (val_result.hardware_result.hardware_id == hw_id and
                        val_result.hardware_result.model_id == mdl_id):
                        relevant_results.append(val_result)
                
                if relevant_results:
                    # Use the most recent validation result
                    relevant_results.sort(key=lambda x: x.validation_timestamp, reverse=True)
                    val_result = relevant_results[0]
                    
                    if metric_name == "all":
                        # Calculate average MAPE across all metrics
                        mape_values = []
                        for metric, comparison in val_result.metrics_comparison.items():
                            if "mape" in comparison:
                                mape_values.append(comparison["mape"])
                        
                        if mape_values:
                            avg_mape = sum(mape_values) / len(mape_values)
                            matrix[i][j] = avg_mape
                    else:
                        # Extract MAPE for the specified metric
                        if metric_name in val_result.metrics_comparison and "mape" in val_result.metrics_comparison[metric_name]:
                            matrix[i][j] = val_result.metrics_comparison[metric_name]["mape"]
        
        # Generate chart title
        if title is None:
            if metric_name == "all":
                title = "Hardware Comparison: Average MAPE Across All Metrics"
            else:
                title = f"Hardware Comparison: MAPE for {metric_name}"
        
        # Create chart
        if pandas_available and numpy_available:
            df = pd.DataFrame(matrix, index=hw_list, columns=mdl_list)
            
            if use_interactive and plotly_available:
                return self._create_interactive_heatmap(df, title, output_path)
            else:
                return self._create_static_heatmap(df, title, output_path)
        else:
            # Basic implementation without pandas/numpy
            if matplotlib_available:
                return self._create_basic_heatmap(matrix, hw_list, mdl_list, title, output_path)
            else:
                logger.error("Cannot create hardware comparison heatmap: required dependencies not available")
                return None
    
    def create_metric_importance_chart(
        self,
        validation_results: List[ValidationResult],
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart showing which metrics have the highest error/importance.
        
        Args:
            validation_results: List of validation results
            hardware_id: Optional hardware ID filter
            model_id: Optional model ID filter
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for metric importance chart")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Filter validation results if needed
        filtered_results = []
        for val_result in validation_results:
            if hardware_id and val_result.hardware_result.hardware_id != hardware_id:
                continue
            if model_id and val_result.hardware_result.model_id != model_id:
                continue
            filtered_results.append(val_result)
        
        if not filtered_results:
            logger.warning("No validation results found for the specified filters")
            return None
        
        # Extract metric importance data
        metrics = set()
        metric_mapes = {}
        
        for val_result in filtered_results:
            for metric, comparison in val_result.metrics_comparison.items():
                if "mape" in comparison:
                    metrics.add(metric)
                    if metric not in metric_mapes:
                        metric_mapes[metric] = []
                    metric_mapes[metric].append(comparison["mape"])
        
        if not metrics:
            logger.warning("No metrics found with MAPE values")
            return None
        
        # Calculate average MAPE for each metric
        metric_avg_mape = {}
        for metric, mapes in metric_mapes.items():
            if mapes:
                metric_avg_mape[metric] = sum(mapes) / len(mapes)
        
        # Sort metrics by average MAPE
        sorted_metrics = sorted(metric_avg_mape.items(), key=lambda x: x[1], reverse=True)
        
        # Generate chart title
        if title is None:
            if hardware_id and model_id:
                title = f"Metric Importance: {model_id} on {hardware_id}"
            elif hardware_id:
                title = f"Metric Importance: All Models on {hardware_id}"
            elif model_id:
                title = f"Metric Importance: {model_id} on All Hardware"
            else:
                title = "Metric Importance: All Models and Hardware"
        
        # Create chart
        data = [{"metric": metric, "avg_mape": avg_mape} for metric, avg_mape in sorted_metrics]
        
        if pandas_available:
            df = pd.DataFrame(data)
            
            if use_interactive and plotly_available:
                return self._create_interactive_metric_importance(df, title, output_path)
            else:
                return self._create_static_metric_importance(df, title, output_path)
        else:
            # Basic implementation without pandas
            if matplotlib_available:
                return self._create_basic_metric_importance(data, title, output_path)
            else:
                logger.error("Cannot create metric importance chart: required dependencies not available")
                return None
    
    def create_error_correlation_matrix(
        self,
        validation_results: List[ValidationResult],
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a correlation matrix showing relationships between errors in different metrics.
        
        Args:
            validation_results: List of validation results
            hardware_id: Optional hardware ID filter
            model_id: Optional model ID filter
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for error correlation matrix")
            return None
        
        # Check if required dependencies are available
        if not pandas_available or not numpy_available:
            logger.error("Cannot create error correlation matrix: pandas and numpy are required")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Filter validation results if needed
        filtered_results = []
        for val_result in validation_results:
            if hardware_id and val_result.hardware_result.hardware_id != hardware_id:
                continue
            if model_id and val_result.hardware_result.model_id != model_id:
                continue
            filtered_results.append(val_result)
        
        if not filtered_results:
            logger.warning("No validation results found for the specified filters")
            return None
        
        # Extract error data
        data = []
        for val_result in filtered_results:
            row = {}
            for metric, comparison in val_result.metrics_comparison.items():
                if "error" in comparison:
                    row[metric] = comparison["error"]
            
            if row:
                data.append(row)
        
        if not data:
            logger.warning("No error data found for correlation matrix")
            return None
        
        # Create correlation matrix
        df = pd.DataFrame(data)
        
        # Drop columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Ensure we have at least two metrics
        if df.shape[1] < 2:
            logger.warning("Need at least two metrics for correlation matrix")
            return None
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Generate chart title
        if title is None:
            if hardware_id and model_id:
                title = f"Error Correlation: {model_id} on {hardware_id}"
            elif hardware_id:
                title = f"Error Correlation: All Models on {hardware_id}"
            elif model_id:
                title = f"Error Correlation: {model_id} on All Hardware"
            else:
                title = "Error Correlation: All Models and Hardware"
        
        # Create chart
        if use_interactive and plotly_available:
            return self._create_interactive_correlation_matrix(corr_matrix, title, output_path)
        else:
            return self._create_static_correlation_matrix(corr_matrix, title, output_path)
    
    def create_comprehensive_dashboard(
        self,
        validation_results: List[ValidationResult],
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        include_sections: Optional[List[str]] = None
    ) -> Union[str, None]:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            validation_results: List of validation results
            hardware_id: Optional hardware ID filter
            model_id: Optional model ID filter
            output_path: Path to save the dashboard
            title: Custom title for the dashboard
            include_sections: List of sections to include (defaults to all)
            
        Returns:
            Path to the saved dashboard if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for comprehensive dashboard")
            return None
        
        # Check if Plotly is available (required for dashboard)
        if not plotly_available:
            logger.error("Cannot create dashboard: Plotly is required")
            return None
        
        # Default sections
        all_sections = [
            "summary",
            "mape_by_hardware",
            "mape_by_model",
            "metric_importance",
            "hardware_heatmap",
            "error_correlation",
            "time_series"
        ]
        
        sections = include_sections or all_sections
        
        # Generate dashboard title
        if title is None:
            if hardware_id and model_id:
                title = f"Simulation Validation Dashboard: {model_id} on {hardware_id}"
            elif hardware_id:
                title = f"Simulation Validation Dashboard: All Models on {hardware_id}"
            elif model_id:
                title = f"Simulation Validation Dashboard: {model_id} on All Hardware"
            else:
                title = "Simulation Validation Dashboard: All Models and Hardware"
        
        # Create dashboard
        try:
            # Create subplots based on sections
            rows = len(sections)
            fig = make_subplots(
                rows=rows, 
                cols=1,
                subplot_titles=[self._get_section_title(section) for section in sections],
                vertical_spacing=0.1
            )
            
            # Add each section
            for i, section in enumerate(sections, 1):
                try:
                    self._add_dashboard_section(fig, i, 1, section, validation_results, hardware_id, model_id)
                except Exception as e:
                    logger.warning(f"Error adding dashboard section '{section}': {e}")
            
            # Update layout
            fig.update_layout(
                title=title,
                height=400 * rows,
                width=self.config["default_width"],
                showlegend=True
            )
            
            # Save or display
            if output_path:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    # Save dashboard
                    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
                    logger.info(f"Dashboard saved to {output_path}")
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving dashboard to {output_path}: {e}")
            
            # Return figure as HTML
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return None
    
    def create_drift_detection_visualization(
        self,
        drift_results: Dict[str, Any],
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a visualization of drift detection results.
        
        Args:
            drift_results: Results from the drift detector
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the visualization
            title: Custom title for the visualization
            
        Returns:
            Path to the saved visualization if output_path is provided, otherwise None
        """
        if not drift_results or "status" not in drift_results or drift_results["status"] != "success":
            logger.warning("Invalid drift results provided for visualization")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Extract drift metrics
        if "drift_metrics" not in drift_results or not drift_results["drift_metrics"]:
            logger.warning("No drift metrics found in drift results")
            return None
        
        drift_metrics = drift_results["drift_metrics"]
        
        # Generate visualization title
        if title is None:
            hardware_type = drift_results.get("hardware_type", "unknown")
            model_type = drift_results.get("model_type", "unknown")
            title = f"Drift Detection: {model_type} on {hardware_type}"
        
        # Create visualization
        if pandas_available:
            # Convert drift metrics to DataFrame
            data = []
            for metric, values in drift_metrics.items():
                row = {"metric": metric}
                row.update(values)
                data.append(row)
            
            df = pd.DataFrame(data)
            
            if use_interactive and plotly_available:
                return self._create_interactive_drift_visualization(df, drift_results, title, output_path)
            else:
                return self._create_static_drift_visualization(df, drift_results, title, output_path)
        else:
            # Basic implementation without pandas
            if matplotlib_available:
                return self._create_basic_drift_visualization(drift_metrics, drift_results, title, output_path)
            else:
                logger.error("Cannot create drift visualization: required dependencies not available")
                return None
    
    def create_calibration_improvement_chart(
        self,
        before_calibration: List[ValidationResult],
        after_calibration: List[ValidationResult],
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart showing the improvement from calibration.
        
        Args:
            before_calibration: Validation results before calibration
            after_calibration: Validation results after calibration
            interactive: Whether to create an interactive plot (overrides config)
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise None
        """
        if not before_calibration or not after_calibration:
            logger.warning("Missing validation results for calibration improvement chart")
            return None
        
        # Determine if we should use interactive plots
        use_interactive = interactive if interactive is not None else self.config["use_interactive"]
        
        # Check if required dependencies are available
        if use_interactive and not plotly_available:
            logger.warning("Plotly not available. Falling back to static plot.")
            use_interactive = False
        
        if not use_interactive and not matplotlib_available:
            logger.warning("Matplotlib not available. Cannot create static plot.")
            return None
        
        # Ensure lists have the same length
        if len(before_calibration) != len(after_calibration):
            logger.warning("Before and after calibration lists must have the same length")
            return None
        
        # Extract calibration improvement data
        data = []
        
        # Common metrics to compare
        common_metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", "power_consumption_w"]
        
        for i in range(len(before_calibration)):
            before = before_calibration[i]
            after = after_calibration[i]
            
            # Ensure the validation results match by hardware/model
            if (before.hardware_result.hardware_id != after.hardware_result.hardware_id or
                before.hardware_result.model_id != after.hardware_result.model_id):
                logger.warning("Mismatched hardware/model IDs in calibration comparison")
                continue
            
            hw_id = before.hardware_result.hardware_id
            model_id = before.hardware_result.model_id
            
            # Compare metrics
            for metric in common_metrics:
                if (metric in before.metrics_comparison and "mape" in before.metrics_comparison[metric] and
                    metric in after.metrics_comparison and "mape" in after.metrics_comparison[metric]):
                    
                    before_mape = before.metrics_comparison[metric]["mape"]
                    after_mape = after.metrics_comparison[metric]["mape"]
                    
                    # Calculate improvement
                    if before_mape > 0:
                        improvement_pct = (before_mape - after_mape) / before_mape * 100
                    else:
                        improvement_pct = 0
                    
                    data.append({
                        "hardware_id": hw_id,
                        "model_id": model_id,
                        "metric": metric,
                        "before_mape": before_mape,
                        "after_mape": after_mape,
                        "improvement_pct": improvement_pct
                    })
        
        if not data:
            logger.warning("No matching metrics found for calibration improvement chart")
            return None
        
        # Generate chart title
        if title is None:
            title = "Calibration Improvement: Before vs After"
        
        # Create chart
        if pandas_available:
            df = pd.DataFrame(data)
            
            if use_interactive and plotly_available:
                return self._create_interactive_calibration_improvement(df, title, output_path)
            else:
                return self._create_static_calibration_improvement(df, title, output_path)
        else:
            # Basic implementation without pandas
            if matplotlib_available:
                return self._create_basic_calibration_improvement(data, title, output_path)
            else:
                logger.error("Cannot create calibration improvement chart: required dependencies not available")
                return None
    
    def create_3d_error_visualization(
        self,
        validation_results: List[ValidationResult],
        metrics: List[str],
        hardware_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a 3D visualization of errors across multiple metrics.
        
        Args:
            validation_results: List of validation results
            metrics: List of metrics to include (must have exactly 3 metrics)
            hardware_ids: List of hardware IDs to include (if None, include all)
            model_ids: List of model IDs to include (if None, include all)
            output_path: Path to save the visualization
            title: Custom title for the visualization
            
        Returns:
            Path to the saved visualization if output_path is provided, otherwise None
        """
        if not validation_results:
            logger.warning("No validation results provided for 3D error visualization")
            return None
        
        # Check if required dependencies are available
        if not plotly_available:
            logger.error("Cannot create 3D visualization: Plotly is required")
            return None
        
        # Ensure we have exactly 3 metrics
        if len(metrics) != 3:
            logger.error("3D visualization requires exactly 3 metrics")
            return None
        
        # Extract error data
        data = []
        for val_result in validation_results:
            hw_id = val_result.hardware_result.hardware_id
            model_id = val_result.hardware_result.model_id
            
            # Skip if not in the specified hardware/model IDs
            if hardware_ids and hw_id not in hardware_ids:
                continue
            if model_ids and model_id not in model_ids:
                continue
            
            # Extract error values for all 3 metrics
            error_values = {}
            for metric in metrics:
                if metric in val_result.metrics_comparison and "mape" in val_result.metrics_comparison[metric]:
                    error_values[metric] = val_result.metrics_comparison[metric]["mape"]
            
            # Only include if we have all 3 metrics
            if len(error_values) == 3:
                data.append({
                    "hardware_id": hw_id,
                    "model_id": model_id,
                    "x": error_values[metrics[0]],
                    "y": error_values[metrics[1]],
                    "z": error_values[metrics[2]],
                    "x_metric": metrics[0],
                    "y_metric": metrics[1],
                    "z_metric": metrics[2]
                })
        
        if not data:
            logger.warning("No complete data points found for 3D visualization")
            return None
        
        # Generate visualization title
        if title is None:
            title = f"3D Error Visualization: {metrics[0]} vs {metrics[1]} vs {metrics[2]}"
        
        # Create visualization
        if pandas_available:
            df = pd.DataFrame(data)
            return self._create_interactive_3d_visualization(df, title, output_path)
        else:
            # Basic implementation without pandas
            return self._create_basic_3d_visualization(data, title, output_path)
    
    # Private implementation methods for visualizations
    def _create_interactive_mape_chart(self, df, title, output_path):
        """Create an interactive MAPE comparison chart using Plotly."""
        try:
            # Create figure
            fig = px.bar(
                df,
                x="hardware_id",
                y="mape",
                color="model_id",
                barmode="group",
                title=title,
                labels={"hardware_id": "Hardware", "mape": "MAPE (%)", "model_id": "Model"},
                height=self.config["default_height"],
                width=self.config["default_width"]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Hardware",
                yaxis_title="MAPE (%)",
                legend_title="Model",
                font=dict(size=self.config["font_size"])
            )
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(df["hardware_id"].unique()) - 0.5,
                y0=5,
                y1=5,
                line=dict(color="green", width=1, dash="dash"),
                name="Excellent (< 5%)"
            )
            
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(df["hardware_id"].unique()) - 0.5,
                y0=10,
                y1=10,
                line=dict(color="gold", width=1, dash="dash"),
                name="Good (< 10%)"
            )
            
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(df["hardware_id"].unique()) - 0.5,
                y0=15,
                y1=15,
                line=dict(color="orange", width=1, dash="dash"),
                name="Acceptable (< 15%)"
            )
            
            # Save or return
            if output_path:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    fig.write_html(output_path)
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving interactive MAPE chart: {e}")
            
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive MAPE chart: {e}")
            return None
    
    def _create_static_mape_chart(self, df, title, output_path):
        """Create a static MAPE comparison chart using Matplotlib."""
        try:
            # Create figure
            plt.figure(figsize=(self.config["default_width"] / 100, self.config["default_height"] / 100), dpi=self.config["dpi"])
            
            # Plot data
            ax = plt.subplot(111)
            df.pivot(index="hardware_id", columns="model_id", values="mape").plot(
                kind="bar",
                ax=ax,
                colormap=self.config["color_scheme"]
            )
            
            # Set labels and title
            plt.xlabel("Hardware")
            plt.ylabel("MAPE (%)")
            plt.title(title, fontsize=self.config["title_font_size"])
            
            # Add threshold lines
            plt.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="Excellent (< 5%)")
            plt.axhline(y=10, color="gold", linestyle="--", alpha=0.7, label="Good (< 10%)")
            plt.axhline(y=15, color="orange", linestyle="--", alpha=0.7, label="Acceptable (< 15%)")
            
            # Add legend
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches="tight", transparent=self.config["transparent_bg"])
                    plt.close()
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving static MAPE chart: {e}")
                    plt.close()
            
            # Return binary image data
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=self.config["dpi"], bbox_inches="tight", transparent=self.config["transparent_bg"])
            plt.close()
            buf.seek(0)
            
            # Convert to base64 for HTML embedding
            import base64
            img_data = base64.b64encode(buf.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{img_data}">'
            
        except Exception as e:
            logger.error(f"Error creating static MAPE chart: {e}")
            return None
    
    def _create_basic_mape_chart(self, data, title, output_path):
        """Create a basic MAPE chart without pandas using Matplotlib."""
        try:
            # Group data by hardware and model
            hw_model_mape = {}
            for item in data:
                hw_id = item["hardware_id"]
                model_id = item["model_id"]
                mape = item["mape"]
                
                if hw_id not in hw_model_mape:
                    hw_model_mape[hw_id] = {}
                
                hw_model_mape[hw_id][model_id] = mape
            
            # Create lists for plotting
            hardware_ids = sorted(hw_model_mape.keys())
            model_ids = sorted(set([model_id for item in data for model_id in hw_model_mape[item["hardware_id"]]]))
            
            # Create figure
            plt.figure(figsize=(self.config["default_width"] / 100, self.config["default_height"] / 100), dpi=self.config["dpi"])
            
            # Set up bar positions
            bar_width = 0.8 / len(model_ids)
            bar_positions = {}
            
            for i, model_id in enumerate(model_ids):
                bar_positions[model_id] = [j - 0.4 + (i + 0.5) * bar_width for j in range(len(hardware_ids))]
            
            # Plot each model as a group of bars
            for model_id in model_ids:
                values = []
                for hw_id in hardware_ids:
                    values.append(hw_model_mape.get(hw_id, {}).get(model_id, float('nan')))
                
                plt.bar(
                    bar_positions[model_id],
                    values,
                    width=bar_width,
                    label=model_id
                )
            
            # Set labels and title
            plt.xlabel("Hardware")
            plt.ylabel("MAPE (%)")
            plt.title(title, fontsize=self.config["title_font_size"])
            
            # Set x-tick positions and labels
            plt.xticks(range(len(hardware_ids)), hardware_ids, rotation=45, ha="right")
            
            # Add threshold lines
            plt.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="Excellent (< 5%)")
            plt.axhline(y=10, color="gold", linestyle="--", alpha=0.7, label="Good (< 10%)")
            plt.axhline(y=15, color="orange", linestyle="--", alpha=0.7, label="Acceptable (< 15%)")
            
            # Add legend
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches="tight", transparent=self.config["transparent_bg"])
                    plt.close()
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving basic MAPE chart: {e}")
                    plt.close()
            
            # Return binary image data
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=self.config["dpi"], bbox_inches="tight", transparent=self.config["transparent_bg"])
            plt.close()
            buf.seek(0)
            
            # Convert to base64 for HTML embedding
            import base64
            img_data = base64.b64encode(buf.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{img_data}">'
            
        except Exception as e:
            logger.error(f"Error creating basic MAPE chart: {e}")
            return None
    
    # Additional private methods for different visualization types
    # Note: Implementation details for all visualization types would be provided here
    
    def _get_section_title(self, section):
        """Get a human-readable title for a dashboard section."""
        titles = {
            "summary": "Summary Statistics",
            "mape_by_hardware": "MAPE by Hardware",
            "mape_by_model": "MAPE by Model",
            "metric_importance": "Metric Importance",
            "hardware_heatmap": "Hardware Comparison Heatmap",
            "error_correlation": "Error Correlation Matrix",
            "time_series": "MAPE Time Series"
        }
        return titles.get(section, section.replace("_", " ").title())
    
    def _add_dashboard_section(self, fig, row, col, section, validation_results, hardware_id, model_id):
        """Add a section to the dashboard figure."""
        try:
            # Filter validation results if hardware_id or model_id specified
            filtered_results = []
            for val_result in validation_results:
                hw_id = val_result.hardware_result.hardware_id
                mdl_id = val_result.hardware_result.model_id
                
                if ((hardware_id is None or hw_id == hardware_id) and
                    (model_id is None or mdl_id == model_id)):
                    filtered_results.append(val_result)
            
            # If no results after filtering, show a message
            if not filtered_results:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    text="No data available for the selected criteria",
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                    row=row,
                    col=col
                )
                return
            
            # Get the most recent results for each hardware-model combination
            latest_results = {}
            for val_result in filtered_results:
                hw_id = val_result.hardware_result.hardware_id
                mdl_id = val_result.hardware_result.model_id
                key = (hw_id, mdl_id)
                
                if key not in latest_results or (
                    val_result.validation_timestamp > latest_results[key].validation_timestamp):
                    latest_results[key] = val_result
            
            # Add the section based on type
            if section == "summary":
                # Summary section with key metrics
                summary_data = []
                
                for (hw_id, mdl_id), val_result in latest_results.items():
                    # Calculate average MAPE across metrics
                    mape_values = []
                    for metric, comparison in val_result.metrics_comparison.items():
                        if "mape" in comparison:
                            mape_values.append(comparison["mape"])
                    
                    if mape_values:
                        avg_mape = sum(mape_values) / len(mape_values)
                        
                        summary_data.append({
                            "hardware_id": hw_id,
                            "model_id": mdl_id,
                            "average_mape": avg_mape,
                            "num_metrics": len(mape_values),
                            "timestamp": val_result.validation_timestamp
                        })
                
                if summary_data:
                    # Create summary table
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = summary_df.sort_values("average_mape")
                    
                    # Create table
                    cell_colors = []
                    for i, row in summary_df.iterrows():
                        mape = row["average_mape"]
                        if mape < 5:
                            color = "rgba(0, 255, 0, 0.1)"  # Green
                        elif mape < 10:
                            color = "rgba(255, 255, 0, 0.1)"  # Yellow
                        elif mape < 15:
                            color = "rgba(255, 165, 0, 0.1)"  # Orange
                        else:
                            color = "rgba(255, 0, 0, 0.1)"  # Red
                        
                        cell_colors.append([color] * len(summary_df.columns))
                    
                    # Format the table
                    summary_df["average_mape"] = summary_df["average_mape"].apply(lambda x: f"{x:.2f}%")
                    summary_df["timestamp"] = pd.to_datetime(summary_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
                    
                    # Add table
                    fig.add_trace(
                        go.Table(
                            header=dict(
                                values=["Hardware", "Model", "Avg MAPE", "# Metrics", "Last Validated"],
                                fill_color="rgba(100, 100, 100, 0.1)",
                                align="left",
                                font=dict(size=12)
                            ),
                            cells=dict(
                                values=[
                                    summary_df["hardware_id"],
                                    summary_df["model_id"],
                                    summary_df["average_mape"],
                                    summary_df["num_metrics"],
                                    summary_df["timestamp"]
                                ],
                                fill_color=cell_colors,
                                align="left",
                                font=dict(size=11)
                            )
                        ),
                        row=row,
                        col=col
                    )
                else:
                    # Show message if no summary data
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="No summary data available",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
            
            elif section == "mape_by_hardware":
                # MAPE by Hardware section
                data = []
                
                for (hw_id, mdl_id), val_result in latest_results.items():
                    # Calculate average MAPE across metrics
                    mape_values = []
                    for metric, comparison in val_result.metrics_comparison.items():
                        if "mape" in comparison:
                            mape_values.append(comparison["mape"])
                    
                    if mape_values:
                        avg_mape = sum(mape_values) / len(mape_values)
                        
                        data.append({
                            "hardware_id": hw_id,
                            "model_id": mdl_id,
                            "mape": avg_mape
                        })
                
                if data:
                    # Create DataFrame
                    df = pd.DataFrame(data)
                    
                    # Create bar chart
                    bar = go.Bar(
                        x=df["hardware_id"],
                        y=df["mape"],
                        marker=dict(
                            color=df["mape"],
                            colorscale=[
                                [0, "green"],
                                [0.33, "yellow"],
                                [0.67, "orange"],
                                [1, "red"]
                            ],
                            colorbar=dict(
                                title="MAPE (%)",
                                thickness=15,
                                len=0.7,
                                y=0.5
                            ),
                            cmin=0,
                            cmax=30  # Cap at 30% for better color range
                        ),
                        text=df["mape"].apply(lambda x: f"{x:.1f}%"),
                        textposition="auto",
                        hovertemplate="Hardware: %{x}<br>MAPE: %{y:.2f}%<extra></extra>"
                    )
                    
                    fig.add_trace(bar, row=row, col=col)
                    
                    # Update axes
                    fig.update_xaxes(title="Hardware", row=row, col=col, tickangle=45)
                    fig.update_yaxes(title="Average MAPE (%)", row=row, col=col)
                    
                    # Add threshold lines
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(df["hardware_id"]) - 0.5,
                        y0=5,
                        y1=5,
                        line=dict(color="green", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(df["hardware_id"]) - 0.5,
                        y0=10,
                        y1=10,
                        line=dict(color="gold", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(df["hardware_id"]) - 0.5,
                        y0=15,
                        y1=15,
                        line=dict(color="orange", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                else:
                    # Show message if no data
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="No hardware data available",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
            
            elif section == "mape_by_model":
                # MAPE by Model section
                data = []
                
                for (hw_id, mdl_id), val_result in latest_results.items():
                    # Calculate average MAPE across metrics
                    mape_values = []
                    for metric, comparison in val_result.metrics_comparison.items():
                        if "mape" in comparison:
                            mape_values.append(comparison["mape"])
                    
                    if mape_values:
                        avg_mape = sum(mape_values) / len(mape_values)
                        
                        data.append({
                            "hardware_id": hw_id,
                            "model_id": mdl_id,
                            "mape": avg_mape
                        })
                
                if data:
                    # Create DataFrame
                    df = pd.DataFrame(data)
                    
                    # Group by model and calculate mean
                    model_grouped = df.groupby("model_id")["mape"].mean().reset_index()
                    model_grouped = model_grouped.sort_values("mape")
                    
                    # Create bar chart
                    bar = go.Bar(
                        x=model_grouped["model_id"],
                        y=model_grouped["mape"],
                        marker=dict(
                            color=model_grouped["mape"],
                            colorscale=[
                                [0, "green"],
                                [0.33, "yellow"],
                                [0.67, "orange"],
                                [1, "red"]
                            ],
                            cmin=0,
                            cmax=30  # Cap at 30% for better color range
                        ),
                        text=model_grouped["mape"].apply(lambda x: f"{x:.1f}%"),
                        textposition="auto",
                        hovertemplate="Model: %{x}<br>MAPE: %{y:.2f}%<extra></extra>"
                    )
                    
                    fig.add_trace(bar, row=row, col=col)
                    
                    # Update axes
                    fig.update_xaxes(title="Model", row=row, col=col, tickangle=45)
                    fig.update_yaxes(title="Average MAPE (%)", row=row, col=col)
                    
                    # Add threshold lines
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(model_grouped["model_id"]) - 0.5,
                        y0=5,
                        y1=5,
                        line=dict(color="green", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(model_grouped["model_id"]) - 0.5,
                        y0=10,
                        y1=10,
                        line=dict(color="gold", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(model_grouped["model_id"]) - 0.5,
                        y0=15,
                        y1=15,
                        line=dict(color="orange", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                else:
                    # Show message if no data
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="No model data available",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
            
            elif section == "metric_importance":
                # Metric Importance section - showing which metrics have highest errors
                all_metrics = {}
                
                for val_result in filtered_results:
                    for metric, comparison in val_result.metrics_comparison.items():
                        if "mape" in comparison:
                            if metric not in all_metrics:
                                all_metrics[metric] = []
                            all_metrics[metric].append(comparison["mape"])
                
                # Calculate average MAPE for each metric
                metric_avg_mape = {}
                for metric, mapes in all_metrics.items():
                    if mapes:
                        metric_avg_mape[metric] = sum(mapes) / len(mapes)
                
                # Sort metrics by average MAPE
                sorted_metrics = sorted(metric_avg_mape.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_metrics:
                    # Create DataFrame
                    metric_data = [{"metric": metric, "avg_mape": avg_mape} for metric, avg_mape in sorted_metrics]
                    df = pd.DataFrame(metric_data)
                    
                    # Create horizontal bar chart
                    bar = go.Bar(
                        y=df["metric"],
                        x=df["avg_mape"],
                        orientation="h",
                        marker=dict(
                            color=df["avg_mape"],
                            colorscale=[
                                [0, "green"],
                                [0.33, "yellow"],
                                [0.67, "orange"],
                                [1, "red"]
                            ],
                            cmin=0,
                            cmax=30  # Cap at 30% for better color range
                        ),
                        text=df["avg_mape"].apply(lambda x: f"{x:.1f}%"),
                        textposition="auto",
                        hovertemplate="Metric: %{y}<br>MAPE: %{x:.2f}%<extra></extra>"
                    )
                    
                    fig.add_trace(bar, row=row, col=col)
                    
                    # Update axes
                    fig.update_xaxes(title="Average MAPE (%)", row=row, col=col)
                    fig.update_yaxes(title="Metric", row=row, col=col)
                    
                    # Add threshold lines
                    fig.add_shape(
                        type="line",
                        y0=-0.5,
                        y1=len(df["metric"]) - 0.5,
                        x0=5,
                        x1=5,
                        line=dict(color="green", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        y0=-0.5,
                        y1=len(df["metric"]) - 0.5,
                        x0=10,
                        x1=10,
                        line=dict(color="gold", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        y0=-0.5,
                        y1=len(df["metric"]) - 0.5,
                        x0=15,
                        x1=15,
                        line=dict(color="orange", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                else:
                    # Show message if no data
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="No metric importance data available",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
            
            elif section == "hardware_heatmap":
                # Hardware Heatmap section
                # Extract unique hardware and model IDs
                hw_ids = set()
                mdl_ids = set()
                
                for val_result in filtered_results:
                    hw_ids.add(val_result.hardware_result.hardware_id)
                    mdl_ids.add(val_result.hardware_result.model_id)
                
                hw_list = sorted(list(hw_ids))
                mdl_list = sorted(list(mdl_ids))
                
                # Initialize matrix with NaN
                if numpy_available:
                    matrix = np.full((len(hw_list), len(mdl_list)), np.nan)
                else:
                    matrix = [[None for _ in range(len(mdl_list))] for _ in range(len(hw_list))]
                
                # Populate matrix with average MAPE values
                for (hw_id, mdl_id), val_result in latest_results.items():
                    if hw_id in hw_list and mdl_id in mdl_list:
                        i = hw_list.index(hw_id)
                        j = mdl_list.index(mdl_id)
                        
                        # Calculate average MAPE across metrics
                        mape_values = []
                        for metric, comparison in val_result.metrics_comparison.items():
                            if "mape" in comparison:
                                mape_values.append(comparison["mape"])
                        
                        if mape_values:
                            avg_mape = sum(mape_values) / len(mape_values)
                            matrix[i][j] = avg_mape
                
                if pandas_available and numpy_available:
                    df = pd.DataFrame(matrix, index=hw_list, columns=mdl_list)
                    
                    # Create heatmap
                    heatmap = go.Heatmap(
                        z=df.values,
                        x=df.columns,
                        y=df.index,
                        colorscale='RdYlGn_r',  # Red (high MAPE) to Green (low MAPE)
                        zmin=0,
                        zmax=30,  # Limit to 30% MAPE for better color differentiation
                        colorbar=dict(
                            title="MAPE (%)",
                            thickness=15,
                            len=0.7,
                            y=0.5
                        ),
                        hoverongaps=False,
                        text=[[f"{val:.2f}%" if not pd.isna(val) else "N/A" for val in row] for row in df.values],
                        hovertemplate="<b>Hardware:</b> %{y}<br>" +
                                    "<b>Model:</b> %{x}<br>" +
                                    "<b>MAPE:</b> %{text}<br>" +
                                    "<extra></extra>"
                    )
                    
                    fig.add_trace(heatmap, row=row, col=col)
                    
                    # Update axes
                    fig.update_xaxes(title="Model", row=row, col=col, tickangle=45)
                    fig.update_yaxes(title="Hardware", row=row, col=col)
                else:
                    # Show message if pandas/numpy not available
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="Heatmap requires pandas and numpy",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
            
            elif section == "error_correlation":
                # Error Correlation Matrix section
                # Extract all error data
                error_data = {}
                
                for val_result in filtered_results:
                    for metric, comparison in val_result.metrics_comparison.items():
                        if "error" in comparison:
                            if metric not in error_data:
                                error_data[metric] = []
                            error_data[metric].append(comparison["error"])
                
                if pandas_available and numpy_available:
                    # Create DataFrame
                    df = pd.DataFrame(error_data)
                    
                    # Drop columns with all NaN values
                    df = df.dropna(axis=1, how='all')
                    
                    # Ensure we have at least two metrics
                    if df.shape[1] >= 2:
                        # Calculate correlation matrix
                        corr_matrix = df.corr()
                        
                        # Create heatmap
                        heatmap = go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            colorscale='RdBu',  # Blue (negative) to Red (positive)
                            zmin=-1,
                            zmax=1,
                            colorbar=dict(
                                title="Correlation",
                                thickness=15,
                                len=0.7,
                                y=0.5
                            ),
                            text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
                            hovertemplate="<b>Metric X:</b> %{y}<br>" +
                                        "<b>Metric Y:</b> %{x}<br>" +
                                        "<b>Correlation:</b> %{text}<br>" +
                                        "<extra></extra>"
                        )
                        
                        fig.add_trace(heatmap, row=row, col=col)
                        
                        # Update axes
                        fig.update_xaxes(title="Metric", row=row, col=col, tickangle=45)
                        fig.update_yaxes(title="Metric", row=row, col=col)
                    else:
                        # Show message if insufficient metrics
                        fig.add_annotation(
                            x=0.5,
                            y=0.5,
                            text="Need at least 2 metrics for correlation matrix",
                            showarrow=False,
                            font=dict(size=14, color="gray"),
                            row=row,
                            col=col
                        )
                else:
                    # Show message if pandas/numpy not available
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="Correlation matrix requires pandas and numpy",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
            
            elif section == "time_series":
                # Time Series section
                if not hardware_id or not model_id:
                    # Show message if no hardware or model specified
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="Time series requires specific hardware and model",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
                    return
                
                # Filter results for specific hardware/model
                time_series_results = []
                for val_result in filtered_results:
                    if (val_result.hardware_result.hardware_id == hardware_id and
                        val_result.hardware_result.model_id == model_id):
                        time_series_results.append(val_result)
                
                if not time_series_results:
                    # Show message if no results
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"No time series data for {model_id} on {hardware_id}",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
                    return
                
                # Sort by timestamp
                time_series_results.sort(key=lambda x: x.validation_timestamp)
                
                # Extract time series data for all metrics
                ts_data = {
                    "timestamp": [val_result.validation_timestamp for val_result in time_series_results]
                }
                
                # Find all metrics
                all_metrics = set()
                for val_result in time_series_results:
                    for metric in val_result.metrics_comparison.keys():
                        all_metrics.add(metric)
                
                # Extract MAPE values for each metric
                for metric in all_metrics:
                    ts_data[metric] = []
                    for val_result in time_series_results:
                        if metric in val_result.metrics_comparison and "mape" in val_result.metrics_comparison[metric]:
                            ts_data[metric].append(val_result.metrics_comparison[metric]["mape"])
                        else:
                            ts_data[metric].append(None)
                
                if pandas_available:
                    # Create DataFrame
                    df = pd.DataFrame(ts_data)
                    
                    # Convert timestamps to datetime
                    try:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    except Exception as e:
                        logger.warning(f"Could not convert timestamps: {e}")
                    
                    # Create time series plot
                    for metric in all_metrics:
                        # Skip if all values are None/NaN
                        if all(pd.isna(val) for val in df[metric]):
                            continue
                        
                        scatter = go.Scatter(
                            x=df["timestamp"],
                            y=df[metric],
                            mode="lines+markers",
                            name=metric,
                            hovertemplate="Timestamp: %{x}<br>MAPE: %{y:.2f}%<extra></extra>"
                        )
                        
                        fig.add_trace(scatter, row=row, col=col)
                    
                    # Update axes
                    fig.update_xaxes(title="Time", row=row, col=col)
                    fig.update_yaxes(title="MAPE (%)", row=row, col=col)
                    
                    # Add threshold lines
                    fig.add_shape(
                        type="line",
                        x0=df["timestamp"].min(),
                        x1=df["timestamp"].max(),
                        y0=5,
                        y1=5,
                        line=dict(color="green", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=df["timestamp"].min(),
                        x1=df["timestamp"].max(),
                        y0=10,
                        y1=10,
                        line=dict(color="gold", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=df["timestamp"].min(),
                        x1=df["timestamp"].max(),
                        y0=15,
                        y1=15,
                        line=dict(color="orange", width=1, dash="dash"),
                        row=row,
                        col=col
                    )
                else:
                    # Show message if pandas not available
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text="Time series requires pandas",
                        showarrow=False,
                        font=dict(size=14, color="gray"),
                        row=row,
                        col=col
                    )
            
            else:
                # Unknown section
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    text=f"Unknown dashboard section: {section}",
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                    row=row,
                    col=col
                )
        
        except Exception as e:
            # Show error message
            logger.error(f"Error adding dashboard section '{section}': {e}")
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text=f"Error: {str(e)}",
                showarrow=False,
                font=dict(size=12, color="red"),
                row=row,
                col=col
            )

    # Additional interactive visualization methods would be implemented here
    
    def _create_interactive_metric_comparison(self, df, title, show_absolute_values, output_path):
        """Create an interactive metric comparison chart using Plotly."""
        try:
            # Melt dataframe for comparison plotting
            df_melt = pd.melt(
                df, 
                id_vars=["metric"], 
                value_vars=["simulation", "hardware"],
                var_name="source",
                value_name="value"
            )
            
            # Create color map
            colors = {"simulation": "#636EFA", "hardware": "#EF553B"}
            
            if show_absolute_values:
                # Create absolute value comparison (bar chart)
                fig = px.bar(
                    df_melt,
                    x="metric",
                    y="value",
                    color="source",
                    barmode="group",
                    title=title,
                    labels={"metric": "Metric", "value": "Value", "source": "Source"},
                    height=self.config["default_height"],
                    width=self.config["default_width"],
                    color_discrete_map=colors
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Value",
                    legend_title="Source",
                    font=dict(size=self.config["font_size"])
                )
                
                # Add percentage difference as text
                for i, row in df.iterrows():
                    metric = row["metric"]
                    sim_val = row["simulation"]
                    hw_val = row["hardware"]
                    
                    if hw_val != 0:
                        pct_diff = (sim_val - hw_val) / hw_val * 100
                        pct_text = f"{pct_diff:.1f}%"
                        
                        # Add annotation above the bars
                        x_pos = i
                        y_pos = max(sim_val, hw_val) * 1.05
                        
                        fig.add_annotation(
                            x=metric,
                            y=y_pos,
                            text=pct_text,
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            bgcolor="rgba(255, 255, 255, 0.7)",
                            bordercolor="rgba(0, 0, 0, 0.3)",
                            borderwidth=1,
                            borderpad=4
                        )
            else:
                # Create normalized comparison (ratio chart)
                df["ratio"] = df["simulation"] / df["hardware"]
                df["error_pct"] = (df["simulation"] - df["hardware"]) / df["hardware"] * 100
                
                # Create figure
                fig = px.bar(
                    df,
                    x="metric",
                    y="error_pct",
                    title=title,
                    labels={"metric": "Metric", "error_pct": "Error (%)"},
                    height=self.config["default_height"],
                    width=self.config["default_width"],
                    color="error_pct",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Error (%)",
                    coloraxis_colorbar=dict(title="Error (%)"),
                    font=dict(size=self.config["font_size"])
                )
                
                # Add reference line at 0%
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(df["metric"]) - 0.5,
                    y0=0,
                    y1=0,
                    line=dict(color="black", width=1, dash="dash")
                )
                
                # Add tolerance bands
                fig.add_shape(
                    type="rect",
                    x0=-0.5,
                    x1=len(df["metric"]) - 0.5,
                    y0=-5,
                    y1=5,
                    fillcolor="rgba(0, 255, 0, 0.1)",
                    line=dict(color="green", width=0),
                    layer="below"
                )
                
                # Add annotations for each bar
                for i, row in df.iterrows():
                    metric = row["metric"]
                    error_pct = row["error_pct"]
                    
                    fig.add_annotation(
                        x=metric,
                        y=error_pct,
                        text=f"{error_pct:.1f}%",
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="rgba(0, 0, 0, 0.3)",
                        borderwidth=1,
                        borderpad=4,
                        yshift=10 if error_pct > 0 else -10
                    )
            
            # Add hover template
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                "Value: %{y}<br>" +
                "<extra></extra>"
            )
            
            # Save or return
            if output_path:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    fig.write_html(output_path)
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving interactive metric comparison chart: {e}")
            
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive metric comparison chart: {e}")
            return None
    
    def _create_static_metric_comparison(self, df, title, show_absolute_values, output_path):
        """Create a static metric comparison chart using Matplotlib."""
        pass
    
    def _create_basic_metric_comparison(self, data, title, show_absolute_values, output_path):
        """Create a basic metric comparison chart without pandas using Matplotlib."""
        pass
    
    def _create_interactive_error_distribution(self, df, title, metric_name, output_path):
        """Create an interactive error distribution chart using Plotly."""
        try:
            # Create subplot with two rows for histogram and box plot
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=["Error Distribution Histogram", "Error Box Plot by Hardware/Model"],
                vertical_spacing=0.2,
                specs=[[{"type": "histogram"}], [{"type": "box"}]]
            )
            
            # Create histogram for error distribution
            fig.add_trace(
                go.Histogram(
                    x=df["error"],
                    name="Error Distribution",
                    marker=dict(
                        color="rgba(99, 110, 250, 0.7)",
                        line=dict(color="rgba(99, 110, 250, 1.0)", width=1)
                    ),
                    hovertemplate="Error: %{x:.2f}<br>Count: %{y}<extra></extra>",
                    nbinsx=30  # Number of bins
                ),
                row=1, col=1
            )
            
            # Add normal distribution curve if we have enough data points and scipy
            if len(df) >= 8 and scipy_available:
                try:
                    # Extract error values and remove NaNs
                    errors = df["error"].dropna().values
                    
                    # Fit normal distribution
                    mu, sigma = stats.norm.fit(errors)
                    
                    # Create x points for the normal curve
                    x = np.linspace(min(errors), max(errors), 100)
                    y = stats.norm.pdf(x, mu, sigma) * len(errors) * (max(errors) - min(errors)) / 30
                    
                    # Add normal curve as scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=x, 
                            y=y,
                            mode="lines",
                            name="Normal Fit",
                            line=dict(color="rgba(255, 0, 0, 0.7)", width=2),
                            hovertemplate="Error: %{x:.2f}<extra>Normal Fit</extra>"
                        ),
                        row=1, col=1
                    )
                    
                    # Add annotation with distribution parameters
                    fig.add_annotation(
                        x=0.95,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"μ = {mu:.2f}<br>σ = {sigma:.2f}",
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="rgba(0, 0, 0, 0.3)",
                        borderwidth=1,
                        borderpad=4,
                        align="right",
                        row=1, col=1
                    )
                except Exception as e:
                    logger.warning(f"Error fitting normal distribution: {e}")
            
            # Create box plot grouped by hardware and model
            if "hardware_id" in df.columns and "model_id" in df.columns:
                # Create combined identifier for hover info
                df["hw_model"] = df["hardware_id"] + " / " + df["model_id"]
                
                # Add box plot
                fig.add_trace(
                    go.Box(
                        x=df["hw_model"],
                        y=df["error"],
                        name="Error by Hardware/Model",
                        marker=dict(color="rgba(99, 110, 250, 0.7)"),
                        hovertemplate="Hardware/Model: %{x}<br>Error: %{y:.2f}<extra></extra>"
                    ),
                    row=2, col=1
                )
            else:
                # Just use hardware_id if that's all we have
                if "hardware_id" in df.columns:
                    fig.add_trace(
                        go.Box(
                            x=df["hardware_id"],
                            y=df["error"],
                            name="Error by Hardware",
                            marker=dict(color="rgba(99, 110, 250, 0.7)"),
                            hovertemplate="Hardware: %{x}<br>Error: %{y:.2f}<extra></extra>"
                        ),
                        row=2, col=1
                    )
                # Or model_id if that's what we have
                elif "model_id" in df.columns:
                    fig.add_trace(
                        go.Box(
                            x=df["model_id"],
                            y=df["error"],
                            name="Error by Model",
                            marker=dict(color="rgba(99, 110, 250, 0.7)"),
                            hovertemplate="Model: %{x}<br>Error: %{y:.2f}<extra></extra>"
                        ),
                        row=2, col=1
                    )
                
            # Update layout
            fig.update_layout(
                title=title,
                width=self.config["default_width"],
                height=int(self.config["default_height"] * 1.5),  # Taller to accommodate two plots
                showlegend=False,
                font=dict(size=self.config["font_size"])
            )
            
            # Update x and y axes for histogram
            fig.update_xaxes(title_text="Error Value", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            
            # Update x and y axes for box plot
            if "hw_model" in df.columns:
                fig.update_xaxes(title_text="Hardware / Model", row=2, col=1, tickangle=45)
            elif "hardware_id" in df.columns:
                fig.update_xaxes(title_text="Hardware", row=2, col=1, tickangle=45)
            elif "model_id" in df.columns:
                fig.update_xaxes(title_text="Model", row=2, col=1, tickangle=45)
            
            fig.update_yaxes(title_text="Error Value", row=2, col=1)
            
            # Add reference line at 0 for histogram
            fig.add_shape(
                type="line",
                x0=0, x1=0,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="black", width=1, dash="dash"),
                row=1, col=1
            )
            
            # Add reference line at 0 for box plot
            fig.add_shape(
                type="line",
                x0=0, x1=1,
                y0=0, y1=0,
                xref="paper",
                line=dict(color="black", width=1, dash="dash"),
                row=2, col=1
            )
            
            # Add MAPE equivalents if we have MAPE values
            if "mape" in df.columns:
                try:
                    # Calculate summary statistics
                    mean_error = df["error"].mean()
                    mean_mape = df["mape"].mean()
                    median_error = df["error"].median()
                    median_mape = df["mape"].median()
                    
                    # Add annotations
                    fig.add_annotation(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=f"Mean Error: {mean_error:.2f}<br>Mean MAPE: {mean_mape:.2f}%<br>Median Error: {median_error:.2f}<br>Median MAPE: {median_mape:.2f}%",
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="rgba(0, 0, 0, 0.3)",
                        borderwidth=1,
                        borderpad=4,
                        align="left"
                    )
                except Exception as e:
                    logger.warning(f"Error calculating MAPE statistics: {e}")
            
            # Save or return
            if output_path:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    fig.write_html(output_path)
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving interactive error distribution chart: {e}")
            
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive error distribution chart: {e}")
            return None
    
    def _create_static_error_distribution(self, df, title, metric_name, output_path):
        """Create a static error distribution chart using Matplotlib."""
        pass
    
    def _create_basic_error_distribution(self, data, title, metric_name, output_path):
        """Create a basic error distribution chart without pandas using Matplotlib."""
        pass
    
    def _create_interactive_time_series(self, df, title, metric_name, show_trend, output_path):
        """Create an interactive time series chart using Plotly."""
        pass
    
    def _create_static_time_series(self, df, title, metric_name, show_trend, output_path):
        """Create a static time series chart using Matplotlib."""
        pass
    
    def _create_basic_time_series(self, data, title, metric_name, show_trend, output_path):
        """Create a basic time series chart without pandas using Matplotlib."""
        pass
    
    def _create_interactive_heatmap(self, df, title, output_path):
        """Create an interactive heatmap using Plotly."""
        try:
            # Create figure with heatmap
            fig = go.Figure(data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale='RdYlGn_r',  # Red (high MAPE) to Green (low MAPE)
                zmin=0,
                zmax=30,  # Limit to 30% MAPE for better color differentiation
                colorbar=dict(
                    title="MAPE (%)",
                    titleside="right",
                    titlefont=dict(size=14),
                    tickfont=dict(size=12),
                ),
                hoverongaps=False,
                text=[[f"{val:.2f}%" if not pd.isna(val) else "N/A" for val in row] for row in df.values],
                hovertemplate="<b>Hardware:</b> %{y}<br>" +
                            "<b>Model:</b> %{x}<br>" +
                            "<b>MAPE:</b> %{text}<br>" +
                            "<extra></extra>"
            ))

            # Update layout
            fig.update_layout(
                title=title,
                width=self.config["default_width"],
                height=self.config["default_height"],
                xaxis=dict(
                    title="Model",
                    tickangle=-45,
                    tickfont=dict(size=12),
                    titlefont=dict(size=14)
                ),
                yaxis=dict(
                    title="Hardware",
                    tickfont=dict(size=12),
                    titlefont=dict(size=14)
                ),
                font=dict(size=self.config["font_size"]),
                margin=dict(t=80, b=80, l=80, r=80),
            )

            # Add text annotations with MAPE values
            annotations = []
            for i, row_idx in enumerate(df.index):
                for j, col_idx in enumerate(df.columns):
                    value = df.loc[row_idx, col_idx]
                    
                    # Only add annotation if we have a value
                    if not pd.isna(value):
                        text_color = "black" if value < 15 else "white"
                        annotations.append(dict(
                            x=col_idx,
                            y=row_idx,
                            text=f"{value:.1f}%",
                            showarrow=False,
                            font=dict(color=text_color, size=10)
                        ))

            fig.update_layout(annotations=annotations)

            # Add color-coded rectangles for MAPE categories
            categories = [
                {"name": "Excellent (<5%)", "color": "rgba(0, 128, 0, 0.1)", "y0": -0.5, "y1": -0.2},
                {"name": "Good (5-10%)", "color": "rgba(255, 255, 0, 0.1)", "y0": -0.8, "y1": -0.5},
                {"name": "Fair (10-15%)", "color": "rgba(255, 165, 0, 0.1)", "y0": -1.1, "y1": -0.8},
                {"name": "Poor (>15%)", "color": "rgba(255, 0, 0, 0.1)", "y0": -1.4, "y1": -1.1}
            ]

            # Add legend shapes and annotations
            for i, cat in enumerate(categories):
                # Add rectangle
                fig.add_shape(
                    type="rect",
                    x0=-0.5,
                    x1=0,
                    y0=cat["y0"],
                    y1=cat["y1"],
                    fillcolor=cat["color"],
                    line=dict(color="rgba(0,0,0,0.3)", width=1),
                    xref="x",
                    yref="y"
                )
                
                # Add label
                fig.add_annotation(
                    x=0.5,
                    y=(cat["y0"] + cat["y1"]) / 2,
                    text=cat["name"],
                    showarrow=False,
                    font=dict(size=10),
                    xref="x",
                    yref="y"
                )

            # Save or return
            if output_path:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    fig.write_html(output_path)
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving interactive heatmap: {e}")
            
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
            
        except Exception as e:
            logger.error(f"Error creating interactive heatmap: {e}")
            return None
    
    def _create_static_heatmap(self, df, title, output_path):
        """Create a static heatmap using Matplotlib."""
        pass
    
    def _create_basic_heatmap(self, matrix, hw_list, mdl_list, title, output_path):
        """Create a basic heatmap without pandas using Matplotlib."""
        pass
    
    def _create_interactive_metric_importance(self, df, title, output_path):
        """Create an interactive metric importance chart using Plotly."""
        pass
    
    def _create_static_metric_importance(self, df, title, output_path):
        """Create a static metric importance chart using Matplotlib."""
        pass
    
    def _create_basic_metric_importance(self, data, title, output_path):
        """Create a basic metric importance chart without pandas using Matplotlib."""
        pass
    
    def _create_interactive_correlation_matrix(self, corr_matrix, title, output_path):
        """Create an interactive correlation matrix using Plotly."""
        pass
    
    def _create_static_correlation_matrix(self, corr_matrix, title, output_path):
        """Create a static correlation matrix using Matplotlib."""
        pass
    
    def _create_interactive_drift_visualization(self, df, drift_results, title, output_path):
        """Create an interactive drift detection visualization using Plotly."""
        pass
    
    def _create_static_drift_visualization(self, df, drift_results, title, output_path):
        """Create a static drift detection visualization using Matplotlib."""
        pass
    
    def _create_basic_drift_visualization(self, drift_metrics, drift_results, title, output_path):
        """Create a basic drift detection visualization without pandas using Matplotlib."""
        pass
    
    def _create_interactive_calibration_improvement(self, df, title, output_path):
        """Create an interactive calibration improvement chart using Plotly."""
        pass
    
    def _create_static_calibration_improvement(self, df, title, output_path):
        """Create a static calibration improvement chart using Matplotlib."""
        pass
    
    def _create_basic_calibration_improvement(self, data, title, output_path):
        """Create a basic calibration improvement chart without pandas using Matplotlib."""
        pass
    
    def _create_interactive_3d_visualization(self, df, title, output_path):
        """Create an interactive 3D visualization using Plotly."""
        pass
    
    def _create_basic_3d_visualization(self, data, title, output_path):
        """Create a basic 3D visualization without pandas using Plotly."""
        pass