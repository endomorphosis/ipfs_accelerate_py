"""
3D Visualization Component for the Advanced Visualization System.

This module provides specialized 3D visualization capabilities for multi-dimensional
data analysis, allowing interactive exploration of complex relationships between
multiple performance metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger("3d_visualization")

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
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors


class Visualization3D(BaseVisualization):
    """
    3D Visualization Component.
    
    This component creates interactive 3D visualizations for exploring 
    multi-dimensional data relationships across hardware types, model families,
    and various performance metrics.
    """
    
    def __init__(self, db_connection=None, theme="light", debug=False):
        """Initialize the 3D visualization component."""
        super().__init__(db_connection, theme, debug)
        
        # Additional configuration specific to 3D visualizations
        self.viz_3d_config = {
            "marker_size": 10,          # Size of markers in 3D space
            "marker_opacity": 0.7,      # Opacity of markers
            "marker_line_width": 1,     # Width of marker borders
            "show_annotations": True,   # Show annotations for data points
            "annotation_size": 10,      # Size of annotations
            "height": 800,              # Height of the plot in pixels
            "width": 1000,              # Width of the plot in pixels
            "camera_position": None,    # Optional fixed camera position
            "enable_animation": False,  # Enable animation capabilities
            "animation_frame": None,    # Column to use for animation frames
            "animation_speed": 1,       # Animation speed (seconds per frame)
            "colormap": "viridis",      # Default colormap
            "show_hover_info": True,    # Show hover information
            "hover_template": None,     # Custom hover template
            "regression_plane": False,  # Show 3D regression plane
            "surface_opacity": 0.7,     # Opacity for surfaces
            "axis_grid": True,          # Show axis grid
            "auto_rotate": False,       # Auto-rotate the plot on load
            "show_projections": False,  # Show projections on walls
            "show_surface": False,      # Show surface instead of scatter
            "surface_contours": False,  # Show contours on surface
            "cluster_points": False,    # Apply clustering to points
            "num_clusters": 3,          # Number of clusters if clustering
            "show_cluster_centroids": True,  # Show cluster centroids
            "show_legend": True,        # Show legend
        }
        
        # Metric names and labels
        self.metric_labels = {
            "throughput": "Throughput (items/sec)",
            "throughput_items_per_second": "Throughput (items/sec)",
            "latency": "Latency (ms)",
            "average_latency_ms": "Latency (ms)",
            "memory": "Memory Usage (MB)",
            "memory_peak_mb": "Memory Usage (MB)",
            "power": "Power Consumption (W)",
            "power_consumption_w": "Power Consumption (W)",
            "efficiency": "Efficiency Score",
            "batch_size": "Batch Size",
            "precision_bits": "Precision (bits)",
            "model_size_mb": "Model Size (MB)",
            "parameter_count": "Parameter Count (M)",
            "accuracy": "Accuracy (%)",
            "inference_time_ms": "Inference Time (ms)",
            "initialization_time_ms": "Initialization Time (ms)",
            "total_operations": "Operations (GOPs)",
        }
        
        logger.info("3D Visualization component initialized")
    
    def create_visualization(self, data=None, **kwargs):
        """
        Create a 3D visualization.
        
        This is a wrapper for the more specific create_3d_visualization method.
        
        Args:
            data: Performance data
            **kwargs: Additional arguments passed to create_3d_visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        return self.create_3d_visualization(data, **kwargs)
    
    def create_3d_visualization(self,
                               data=None,
                               x_metric="throughput",
                               y_metric="memory",
                               z_metric="latency",
                               color_by="hardware_type",
                               size_by=None,
                               text_column=None,
                               model_families=None,
                               hardware_types=None,
                               filter_dict=None,
                               output_path=None,
                               title=None,
                               **kwargs):
        """
        Create an interactive 3D visualization for exploring multi-dimensional data.
        
        Args:
            data: Performance data (DataFrame, dict, or path to file)
            x_metric: Metric to display on x-axis
            y_metric: Metric to display on y-axis
            z_metric: Metric to display on z-axis
            color_by: Column to use for coloring points
            size_by: Optional column to use for sizing points
            text_column: Optional column to use for point labels
            model_families: Optional list of model families to include
            hardware_types: Optional list of hardware types to include
            filter_dict: Optional dictionary of {column: value} pairs for filtering
            output_path: Optional path for saving the visualization
            title: Custom title for the visualization
            **kwargs: Additional configuration parameters
            
        Returns:
            Path to the saved 3D visualization, or None if creation failed
        """
        # Update configuration with any provided kwargs
        viz_3d_config = self.viz_3d_config.copy()
        viz_3d_config.update(kwargs)
        
        # Determine visualization title
        if title is None:
            title = f"3D Visualization: {self._get_metric_label(x_metric)} vs {self._get_metric_label(y_metric)} vs {self._get_metric_label(z_metric)}"
        
        # Determine column names for metrics
        x_column = self._get_metric_column(x_metric)
        y_column = self._get_metric_column(y_metric)
        z_column = self._get_metric_column(z_metric)
        
        # Load data if provided or use placeholder
        if data is None:
            # Create sample data for demonstration purposes
            # In a real implementation, this would fetch from the database
            if self.debug:
                logger.info("No data provided, using sample data for demonstration")
            
            # Sample data with hardware types and model families
            hw_list = hardware_types or ["CPU", "GPU", "WebGPU", "WebNN", "MPS"]
            model_list = ["BERT", "ViT", "LLAMA", "Whisper", "CLIP", "T5", "Stable-Diffusion"]
            family_list = model_families or ["Text", "Vision", "Audio", "Multimodal"]
            precision_options = [16, 8, 4, 2]  # FP16, INT8, INT4, INT2
            batch_sizes = [1, 2, 4, 8, 16, 32]
            
            # Create sample dataframe
            rows = []
            for family in family_list:
                # Select models for this family
                if family == "Text":
                    models = ["BERT", "LLAMA", "T5"]
                elif family == "Vision":
                    models = ["ViT", "CLIP"]
                elif family == "Audio":
                    models = ["Whisper"]
                else:
                    models = ["Stable-Diffusion", "CLIP"]
                
                for model in models:
                    for hw in hw_list:
                        # Generate data for different batch sizes and precisions
                        for batch in batch_sizes:
                            for precision in precision_options:
                                # Base values depend on hardware, model, batch size, and precision
                                throughput_base = 100 if hw in ["GPU", "WebGPU"] else 50
                                # Adjust for batch size (larger batches = more throughput)
                                throughput_base *= np.sqrt(batch) / 2
                                # Adjust for precision (lower precision = more throughput)
                                throughput_base *= (32 / max(precision, 1)) / 2
                                
                                # Latency increases with batch size but at a sublinear rate
                                latency_base = 20 if hw in ["GPU", "WebGPU"] else 40
                                latency_base *= np.sqrt(batch) / 2
                                # Lower precision = lower latency
                                latency_base *= precision / 16
                                
                                # Memory usage increases with batch size and precision
                                memory_base = 500 if model in ["LLAMA", "Stable-Diffusion"] else 200
                                memory_base *= batch / 4  # Linear with batch size
                                memory_base *= precision / 8  # Linear with precision bits
                                
                                # Power consumption
                                power_base = 20 if hw in ["GPU"] else 10
                                power_base *= batch / 8
                                power_base *= precision / 8
                                
                                # Model size depends on the model and precision
                                model_size_base = 1000 if model in ["LLAMA", "Stable-Diffusion"] else 300
                                model_size_base *= precision / 16  # Linear with precision
                                
                                # Add randomness
                                throughput = throughput_base * (1 + np.random.normal(0, 0.1))
                                latency = latency_base * (1 + np.random.normal(0, 0.1))
                                memory = memory_base * (1 + np.random.normal(0, 0.05))
                                power = power_base * (1 + np.random.normal(0, 0.1))
                                model_size = model_size_base * (1 + np.random.normal(0, 0.05))
                                
                                # Generate row
                                rows.append({
                                    "model_name": model,
                                    "model_family": family,
                                    "hardware_type": hw,
                                    "batch_size": batch,
                                    "precision_bits": precision,
                                    "throughput_items_per_second": throughput,
                                    "average_latency_ms": latency,
                                    "memory_peak_mb": memory,
                                    "power_consumption_w": power,
                                    "model_size_mb": model_size,
                                    "parameter_count": model_size_base / 4,  # Approximation
                                    "efficiency": throughput / (power + 0.1),  # Throughput per watt
                                    "is_simulated": np.random.random() > 0.7  # Random simulation flag
                                })
            
            df = pd.DataFrame(rows)
        else:
            # Load data from the provided source
            df = self.load_data(data)
            
            if df.empty:
                logger.error("Failed to load data for visualization")
                return None
        
        # Apply filters
        if model_families and "model_family" in df.columns:
            df = df[df["model_family"].isin(model_families)]
        
        if hardware_types and "hardware_type" in df.columns:
            df = df[df["hardware_type"].isin(hardware_types)]
        
        if filter_dict:
            for col, value in filter_dict.items():
                if col in df.columns:
                    if isinstance(value, list):
                        df = df[df[col].isin(value)]
                    else:
                        df = df[df[col] == value]
        
        # Check if required columns are available
        required_columns = [x_column, y_column, z_column]
        if color_by:
            required_columns.append(color_by)
        if size_by:
            required_columns.append(size_by)
        if text_column:
            required_columns.append(text_column)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Create the 3D visualization based on available libraries
        if PLOTLY_AVAILABLE:
            return self._create_interactive_3d(
                df, x_column, y_column, z_column, color_by, size_by, text_column,
                output_path, title, viz_3d_config
            )
        elif MATPLOTLIB_AVAILABLE:
            return self._create_static_3d(
                df, x_column, y_column, z_column, color_by, size_by, text_column,
                output_path, title, viz_3d_config
            )
        else:
            logger.error("Neither Plotly nor Matplotlib is available. Cannot create visualization.")
            return None
    
    def _get_metric_column(self, metric):
        """
        Map a metric name to the corresponding column name.
        
        Args:
            metric: Short metric name
            
        Returns:
            Column name in the dataframe
        """
        # Direct mapping
        if metric == "throughput":
            return "throughput_items_per_second"
        elif metric == "latency":
            return "average_latency_ms"
        elif metric == "memory":
            return "memory_peak_mb"
        elif metric == "power":
            return "power_consumption_w"
        
        # If not found in mapping, return the metric name as is
        # (assuming it might be a direct column name)
        return metric
    
    def _get_metric_label(self, metric):
        """
        Get a human-readable label for a metric.
        
        Args:
            metric: Metric name or column name
            
        Returns:
            Human-readable label for the metric
        """
        # Check direct metric name
        if metric in self.metric_labels:
            return self.metric_labels[metric]
        
        # Check mapped column name
        column = self._get_metric_column(metric)
        if column in self.metric_labels:
            return self.metric_labels[column]
        
        # Default: format the metric name
        return metric.replace("_", " ").title()
    
    def _create_interactive_3d(self, df, x_column, y_column, z_column, 
                              color_by, size_by, text_column,
                              output_path, title, viz_3d_config):
        """
        Create an interactive 3D visualization using Plotly.
        
        Args:
            df: DataFrame with performance data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            color_by: Column to use for coloring points
            size_by: Optional column to use for sizing points
            text_column: Optional column to use for point labels
            output_path: Path to save the visualization
            title: Title for the visualization
            viz_3d_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Determine how to create the visualization based on configuration
            if viz_3d_config.get("enable_animation") and viz_3d_config.get("animation_frame") in df.columns:
                # Create animated 3D plot
                return self._create_animated_3d_plot(
                    df, x_column, y_column, z_column, color_by, size_by, text_column,
                    output_path, title, viz_3d_config
                )
            elif viz_3d_config.get("show_surface"):
                # Create 3D surface plot
                return self._create_3d_surface_plot(
                    df, x_column, y_column, z_column, color_by,
                    output_path, title, viz_3d_config
                )
            elif viz_3d_config.get("cluster_points"):
                # Create 3D scatter plot with clustering
                return self._create_3d_clustered_plot(
                    df, x_column, y_column, z_column, color_by, size_by, text_column,
                    output_path, title, viz_3d_config
                )
            else:
                # Create standard 3D scatter plot
                return self._create_3d_scatter_plot(
                    df, x_column, y_column, z_column, color_by, size_by, text_column,
                    output_path, title, viz_3d_config
                )
        except Exception as e:
            logger.error(f"Error creating interactive 3D visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_3d_scatter_plot(self, df, x_column, y_column, z_column,
                               color_by, size_by, text_column,
                               output_path, title, viz_3d_config):
        """
        Create a standard 3D scatter plot using Plotly.
        
        Args:
            df: DataFrame with performance data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            color_by: Column to use for coloring points
            size_by: Optional column to use for sizing points
            text_column: Optional column to use for point labels
            output_path: Path to save the visualization
            title: Title for the visualization
            viz_3d_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        # Create figure
        fig = go.Figure()
        
        # Prepare hover text
        if viz_3d_config.get("show_hover_info"):
            if viz_3d_config.get("hover_template"):
                hover_template = viz_3d_config["hover_template"]
            else:
                # Generate hover text based on available columns
                hover_parts = []
                
                # Add model name and family if available
                if "model_name" in df.columns:
                    hover_parts.append("<b>Model:</b> %{customdata[0]}")
                if "model_family" in df.columns:
                    hover_parts.append("<b>Family:</b> %{customdata[1]}")
                
                # Add hardware type if available
                if "hardware_type" in df.columns:
                    hover_parts.append("<b>Hardware:</b> %{customdata[2]}")
                
                # Add batch size and precision if available
                if "batch_size" in df.columns:
                    hover_parts.append("<b>Batch Size:</b> %{customdata[3]}")
                if "precision_bits" in df.columns:
                    hover_parts.append("<b>Precision:</b> %{customdata[4]} bits")
                
                # Add axis metrics
                hover_parts.append(f"<b>{self._get_metric_label(x_column)}:</b> %{{x:.2f}}")
                hover_parts.append(f"<b>{self._get_metric_label(y_column)}:</b> %{{y:.2f}}")
                hover_parts.append(f"<b>{self._get_metric_label(z_column)}:</b> %{{z:.2f}}")
                
                # Add custom text if available
                if text_column and text_column in df.columns:
                    hover_parts.append("<b>%{text}</b>")
                
                # Create hover template
                hover_template = "<br>".join(hover_parts)
        else:
            hover_template = None
        
        # Prepare custom data for hover information
        customdata_cols = []
        if "model_name" in df.columns:
            customdata_cols.append(df["model_name"])
        if "model_family" in df.columns:
            customdata_cols.append(df["model_family"])
        if "hardware_type" in df.columns:
            customdata_cols.append(df["hardware_type"])
        if "batch_size" in df.columns:
            customdata_cols.append(df["batch_size"])
        if "precision_bits" in df.columns:
            customdata_cols.append(df["precision_bits"])
        
        customdata = np.stack(customdata_cols, axis=1) if customdata_cols else None
        
        # Determine marker size
        marker_size = viz_3d_config["marker_size"]
        if size_by and size_by in df.columns:
            # Normalize size column to a reasonable range
            size_min = df[size_by].min()
            size_max = df[size_by].max()
            if size_min != size_max:  # Avoid division by zero
                size_norm = (df[size_by] - size_min) / (size_max - size_min)
                # Scale to desired size range
                marker_size = 5 + size_norm * 20
            else:
                marker_size = 10
        
        # Determine coloring
        colorscale = viz_3d_config.get("colormap", "viridis")
        
        # Continuous vs. categorical coloring
        if color_by and color_by in df.columns:
            unique_values = df[color_by].nunique()
            # If few unique values or string type, treat as categorical
            if unique_values <= 10 or pd.api.types.is_string_dtype(df[color_by]):
                # Create a trace for each category
                categories = df[color_by].unique()
                
                # Get a color palette
                if len(categories) <= 10:
                    color_palette = px.colors.qualitative.Plotly
                else:
                    # Generate colors from colorscale
                    import plotly.colors
                    color_palette = plotly.colors.sample_colorscale(
                        colorscale, len(categories))
                
                # Create a trace for each category
                for i, category in enumerate(sorted(categories)):
                    category_df = df[df[color_by] == category]
                    
                    # Get color for this category
                    color = color_palette[i % len(color_palette)]
                    
                    # Get text for this category
                    if text_column and text_column in category_df.columns:
                        text = category_df[text_column]
                    else:
                        text = None
                    
                    # Get customdata for this category
                    if customdata is not None:
                        category_customdata = customdata[df[color_by] == category]
                    else:
                        category_customdata = None
                    
                    # Get marker size for this category
                    if size_by and size_by in df.columns:
                        category_marker_size = marker_size[df[color_by] == category]
                    else:
                        category_marker_size = marker_size
                    
                    # Add trace
                    fig.add_trace(go.Scatter3d(
                        x=category_df[x_column],
                        y=category_df[y_column],
                        z=category_df[z_column],
                        mode='markers',
                        marker=dict(
                            size=category_marker_size,
                            color=color,
                            line=dict(
                                width=viz_3d_config["marker_line_width"],
                                color='white'
                            ),
                            opacity=viz_3d_config["marker_opacity"]
                        ),
                        name=str(category),
                        text=text,
                        customdata=category_customdata,
                        hovertemplate=hover_template
                    ))
            else:
                # Continuous coloring
                if text_column and text_column in df.columns:
                    text = df[text_column]
                else:
                    text = None
                
                fig.add_trace(go.Scatter3d(
                    x=df[x_column],
                    y=df[y_column],
                    z=df[z_column],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=df[color_by],
                        colorscale=colorscale,
                        colorbar=dict(
                            title=self._get_metric_label(color_by)
                        ),
                        line=dict(
                            width=viz_3d_config["marker_line_width"],
                            color='white'
                        ),
                        opacity=viz_3d_config["marker_opacity"]
                    ),
                    name=color_by,
                    text=text,
                    customdata=customdata,
                    hovertemplate=hover_template
                ))
        else:
            # Simple single color 3D scatter
            if text_column and text_column in df.columns:
                text = df[text_column]
            else:
                text = None
            
            fig.add_trace(go.Scatter3d(
                x=df[x_column],
                y=df[y_column],
                z=df[z_column],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=self.theme_colors.get("accent1", "#1f77b4"),
                    line=dict(
                        width=viz_3d_config["marker_line_width"],
                        color='white'
                    ),
                    opacity=viz_3d_config["marker_opacity"]
                ),
                name='Data Points',
                text=text,
                customdata=customdata,
                hovertemplate=hover_template
            ))
        
        # Add regression plane if requested
        if viz_3d_config.get("regression_plane") and len(df) >= 10:
            self._add_regression_plane(fig, df, x_column, y_column, z_column, viz_3d_config)
        
        # Add projections if requested
        if viz_3d_config.get("show_projections"):
            self._add_3d_projections(fig, df, x_column, y_column, z_column, viz_3d_config)
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=self._get_metric_label(x_column),
                yaxis_title=self._get_metric_label(y_column),
                zaxis_title=self._get_metric_label(z_column),
                xaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                yaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                zaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                camera=viz_3d_config.get("camera_position")
            ),
            width=viz_3d_config["width"],
            height=viz_3d_config["height"],
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)" if self.theme == "light" else "rgba(0, 0, 0, 0.5)"
            ),
            autosize=True
        )
        
        # Configure auto-rotation if requested
        if viz_3d_config.get("auto_rotate"):
            scene_aspect = dict(fig.layout.scene.camera) if fig.layout.scene.camera else {}
            fig.layout.updatemenus = [
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Rotate',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=30, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0)
                            )]
                        )
                    ],
                    x=0.1,
                    y=0,
                    direction='left',
                    pad=dict(r=10, t=10),
                    bgcolor='rgba(0,0,0,0.2)'
                )
            ]
            
            # Create frames for rotation animation
            frames = []
            for t in np.linspace(0, 2*np.pi, 60):
                eye_x = 1.5 * np.cos(t)
                eye_y = 1.5 * np.sin(t)
                eye_z = 0.8
                frames.append(
                    go.Frame(
                        layout=dict(
                            scene_camera_eye=dict(
                                x=eye_x, y=eye_y, z=eye_z
                            )
                        )
                    )
                )
            fig.frames = frames
        
        # Export visualization or show it
        if output_path:
            self.figure = fig
            self.export(output_path, format="html")
            return output_path
        else:
            self.figure = fig
            return self.show()
    
    def _create_3d_surface_plot(self, df, x_column, y_column, z_column, color_by,
                               output_path, title, viz_3d_config):
        """
        Create a 3D surface plot using Plotly.
        
        This method creates a surface plot from scattered data points by
        interpolating a 2D grid of values.
        
        Args:
            df: DataFrame with performance data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            color_by: Column to use for coloring (optional)
            output_path: Path to save the visualization
            title: Title for the visualization
            viz_3d_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        # Data must be on a grid for surface plots
        # Convert scattered data points to a gridded format using interpolation
        try:
            from scipy.interpolate import griddata
            
            # Extract data points
            x = df[x_column].values
            y = df[y_column].values
            z = df[z_column].values
            
            # Create grid for interpolation
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate z values on the grid
            zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
            
            # Create figure with surface plot
            fig = go.Figure(data=[
                go.Surface(
                    x=xi,
                    y=yi,
                    z=zi_grid,
                    colorscale=viz_3d_config.get("colormap", "viridis"),
                    opacity=viz_3d_config.get("surface_opacity", 0.7),
                    contours=dict(
                        x=dict(show=viz_3d_config.get("surface_contours", False), width=1.5),
                        y=dict(show=viz_3d_config.get("surface_contours", False), width=1.5),
                        z=dict(show=viz_3d_config.get("surface_contours", False), width=1.5)
                    ),
                    colorbar=dict(
                        title=self._get_metric_label(z_column),
                        thickness=20
                    ),
                    hoverinfo="x+y+z",
                    hovertemplate=f"<b>{self._get_metric_label(x_column)}</b>: %{{x:.2f}}<br>" +
                                f"<b>{self._get_metric_label(y_column)}</b>: %{{y:.2f}}<br>" +
                                f"<b>{self._get_metric_label(z_column)}</b>: %{{z:.2f}}"
                )
            ])
            
            # Add original data points as scatter
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color='black',
                    opacity=0.5
                ),
                name='Data Points',
                showlegend=True
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=self._get_metric_label(x_column),
                    yaxis_title=self._get_metric_label(y_column),
                    zaxis_title=self._get_metric_label(z_column),
                    xaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    yaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    zaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    camera=viz_3d_config.get("camera_position")
                ),
                width=viz_3d_config["width"],
                height=viz_3d_config["height"],
                template="plotly_white" if self.theme == "light" else "plotly_dark",
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            # Export visualization or show it
            if output_path:
                self.figure = fig
                self.export(output_path, format="html")
                return output_path
            else:
                self.figure = fig
                return self.show()
        
        except ImportError:
            logger.warning("SciPy is required for surface plotting. Install with: pip install scipy")
            logger.info("Falling back to scatter plot")
            return self._create_3d_scatter_plot(
                df, x_column, y_column, z_column, color_by, None, None,
                output_path, title, viz_3d_config
            )
        except Exception as e:
            logger.error(f"Error creating 3D surface plot: {e}")
            logger.info("Falling back to scatter plot")
            return self._create_3d_scatter_plot(
                df, x_column, y_column, z_column, color_by, None, None,
                output_path, title, viz_3d_config
            )
    
    def _create_3d_clustered_plot(self, df, x_column, y_column, z_column,
                                 color_by, size_by, text_column,
                                 output_path, title, viz_3d_config):
        """
        Create a 3D scatter plot with automatic clustering of points.
        
        Args:
            df: DataFrame with performance data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            color_by: Column to use for coloring points (ignored in clustering)
            size_by: Optional column to use for sizing points
            text_column: Optional column to use for point labels
            output_path: Path to save the visualization
            title: Title for the visualization
            viz_3d_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Extract features for clustering
            features = df[[x_column, y_column, z_column]].copy()
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Perform clustering
            n_clusters = viz_3d_config.get("num_clusters", 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Add cluster labels to the dataframe
            df = df.copy()  # Create a copy to avoid modifying the original
            df['cluster'] = clusters
            
            # Create figure
            fig = go.Figure()
            
            # Prepare hover text
            if viz_3d_config.get("show_hover_info"):
                if viz_3d_config.get("hover_template"):
                    hover_template = viz_3d_config["hover_template"]
                else:
                    # Generate hover text based on available columns
                    hover_parts = []
                    
                    # Add model name and family if available
                    if "model_name" in df.columns:
                        hover_parts.append("<b>Model:</b> %{customdata[0]}")
                    if "model_family" in df.columns:
                        hover_parts.append("<b>Family:</b> %{customdata[1]}")
                    
                    # Add hardware type if available
                    if "hardware_type" in df.columns:
                        hover_parts.append("<b>Hardware:</b> %{customdata[2]}")
                    
                    # Add batch size and precision if available
                    if "batch_size" in df.columns:
                        hover_parts.append("<b>Batch Size:</b> %{customdata[3]}")
                    if "precision_bits" in df.columns:
                        hover_parts.append("<b>Precision:</b> %{customdata[4]} bits")
                    
                    # Add cluster information
                    hover_parts.append("<b>Cluster:</b> %{marker.color}")
                    
                    # Add axis metrics
                    hover_parts.append(f"<b>{self._get_metric_label(x_column)}:</b> %{{x:.2f}}")
                    hover_parts.append(f"<b>{self._get_metric_label(y_column)}:</b> %{{y:.2f}}")
                    hover_parts.append(f"<b>{self._get_metric_label(z_column)}:</b> %{{z:.2f}}")
                    
                    # Add custom text if available
                    if text_column and text_column in df.columns:
                        hover_parts.append("<b>%{text}</b>")
                    
                    # Create hover template
                    hover_template = "<br>".join(hover_parts)
            else:
                hover_template = None
            
            # Prepare custom data for hover information
            customdata_cols = []
            if "model_name" in df.columns:
                customdata_cols.append(df["model_name"])
            if "model_family" in df.columns:
                customdata_cols.append(df["model_family"])
            if "hardware_type" in df.columns:
                customdata_cols.append(df["hardware_type"])
            if "batch_size" in df.columns:
                customdata_cols.append(df["batch_size"])
            if "precision_bits" in df.columns:
                customdata_cols.append(df["precision_bits"])
            
            customdata = np.stack(customdata_cols, axis=1) if customdata_cols else None
            
            # Determine marker size
            marker_size = viz_3d_config["marker_size"]
            if size_by and size_by in df.columns:
                # Normalize size column to a reasonable range
                size_min = df[size_by].min()
                size_max = df[size_by].max()
                if size_min != size_max:  # Avoid division by zero
                    size_norm = (df[size_by] - size_min) / (size_max - size_min)
                    # Scale to desired size range
                    marker_size = 5 + size_norm * 20
                else:
                    marker_size = 10
            
            # Get color palette for clusters
            color_palette = px.colors.qualitative.Plotly
            # Ensure enough colors
            while len(color_palette) < n_clusters:
                color_palette = color_palette * 2
            
            # Plot clusters
            for cluster_id in range(n_clusters):
                cluster_df = df[df['cluster'] == cluster_id]
                
                # Skip if empty
                if cluster_df.empty:
                    continue
                
                # Get text for this cluster
                if text_column and text_column in cluster_df.columns:
                    text = cluster_df[text_column]
                else:
                    text = None
                
                # Get customdata for this cluster
                if customdata is not None:
                    cluster_customdata = customdata[df['cluster'] == cluster_id]
                else:
                    cluster_customdata = None
                
                # Get marker size for this cluster
                if size_by and size_by in df.columns:
                    cluster_marker_size = marker_size[df['cluster'] == cluster_id]
                else:
                    cluster_marker_size = marker_size
                
                # Add trace for this cluster
                fig.add_trace(go.Scatter3d(
                    x=cluster_df[x_column],
                    y=cluster_df[y_column],
                    z=cluster_df[z_column],
                    mode='markers',
                    marker=dict(
                        size=cluster_marker_size,
                        color=color_palette[cluster_id % len(color_palette)],
                        line=dict(
                            width=viz_3d_config["marker_line_width"],
                            color='white'
                        ),
                        opacity=viz_3d_config["marker_opacity"]
                    ),
                    name=f'Cluster {cluster_id + 1}',
                    text=text,
                    customdata=cluster_customdata,
                    hovertemplate=hover_template
                ))
            
            # Add cluster centroids if requested
            if viz_3d_config.get("show_cluster_centroids"):
                # Inverse transform to get centroids in original scale
                centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                
                # Add trace for centroids
                fig.add_trace(go.Scatter3d(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    z=centroids[:, 2],
                    mode='markers',
                    marker=dict(
                        size=marker_size * 1.5 if isinstance(marker_size, (int, float)) else 15,
                        color=color_palette[:n_clusters],
                        symbol='diamond',
                        line=dict(
                            width=2,
                            color='black'
                        ),
                        opacity=1.0
                    ),
                    name='Centroids',
                    hovertemplate=f"<b>Centroid</b><br>" +
                                  f"<b>{self._get_metric_label(x_column)}</b>: %{{x:.2f}}<br>" +
                                  f"<b>{self._get_metric_label(y_column)}</b>: %{{y:.2f}}<br>" +
                                  f"<b>{self._get_metric_label(z_column)}</b>: %{{z:.2f}}"
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=self._get_metric_label(x_column),
                    yaxis_title=self._get_metric_label(y_column),
                    zaxis_title=self._get_metric_label(z_column),
                    xaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    yaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    zaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    camera=viz_3d_config.get("camera_position")
                ),
                width=viz_3d_config["width"],
                height=viz_3d_config["height"],
                template="plotly_white" if self.theme == "light" else "plotly_dark",
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.5)" if self.theme == "light" else "rgba(0, 0, 0, 0.5)"
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
                
        except ImportError:
            logger.warning("scikit-learn is required for clustering. Install with: pip install scikit-learn")
            logger.info("Falling back to regular scatter plot")
            return self._create_3d_scatter_plot(
                df, x_column, y_column, z_column, color_by, size_by, text_column,
                output_path, title, viz_3d_config
            )
        except Exception as e:
            logger.error(f"Error creating clustered 3D plot: {e}")
            logger.info("Falling back to regular scatter plot")
            return self._create_3d_scatter_plot(
                df, x_column, y_column, z_column, color_by, size_by, text_column,
                output_path, title, viz_3d_config
            )
    
    def _create_animated_3d_plot(self, df, x_column, y_column, z_column,
                                color_by, size_by, text_column,
                                output_path, title, viz_3d_config):
        """
        Create an animated 3D scatter plot using Plotly.
        
        Args:
            df: DataFrame with performance data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            color_by: Column to use for coloring points
            size_by: Optional column to use for sizing points
            text_column: Optional column to use for point labels
            output_path: Path to save the visualization
            title: Title for the visualization
            viz_3d_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        # Use Plotly Express for animated plots
        try:
            animation_frame = viz_3d_config.get("animation_frame")
            if animation_frame not in df.columns:
                logger.error(f"Animation frame column {animation_frame} not found in DataFrame")
                return self._create_3d_scatter_plot(
                    df, x_column, y_column, z_column, color_by, size_by, text_column,
                    output_path, title, viz_3d_config
                )
            
            # Determine hover data columns
            hover_data = []
            if "model_name" in df.columns:
                hover_data.append("model_name")
            if "model_family" in df.columns:
                hover_data.append("model_family")
            if "hardware_type" in df.columns:
                hover_data.append("hardware_type")
            if "batch_size" in df.columns and animation_frame != "batch_size":
                hover_data.append("batch_size")
            if "precision_bits" in df.columns and animation_frame != "precision_bits":
                hover_data.append("precision_bits")
            
            # Create animated plot
            fig = px.scatter_3d(
                df,
                x=x_column,
                y=y_column,
                z=z_column,
                color=color_by if color_by else None,
                size=size_by if size_by else None,
                text=text_column if text_column else None,
                animation_frame=animation_frame,
                animation_group="model_name" if "model_name" in df.columns else None,
                hover_data=hover_data,
                color_continuous_scale=viz_3d_config.get("colormap", "viridis"),
                size_max=viz_3d_config.get("marker_size", 10) * 2,
                opacity=viz_3d_config.get("marker_opacity", 0.7),
                title=title,
                labels={
                    x_column: self._get_metric_label(x_column),
                    y_column: self._get_metric_label(y_column),
                    z_column: self._get_metric_label(z_column),
                    color_by: self._get_metric_label(color_by) if color_by else "",
                    size_by: self._get_metric_label(size_by) if size_by else "",
                    animation_frame: self._get_metric_label(animation_frame)
                }
            )
            
            # Update layout
            fig.update_layout(
                scene=dict(
                    xaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    yaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    zaxis=dict(gridcolor='lightgrey' if viz_3d_config["axis_grid"] else None),
                    camera=viz_3d_config.get("camera_position")
                ),
                width=viz_3d_config["width"],
                height=viz_3d_config["height"],
                template="plotly_white" if self.theme == "light" else "plotly_dark",
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.5)" if self.theme == "light" else "rgba(0, 0, 0, 0.5)"
                )
            )
            
            # Configure animation
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = viz_3d_config.get("animation_speed", 1) * 1000
            
            # Export visualization or show it
            if output_path:
                self.figure = fig
                self.export(output_path, format="html")
                return output_path
            else:
                self.figure = fig
                return self.show()
                
        except Exception as e:
            logger.error(f"Error creating animated 3D plot: {e}")
            logger.info("Falling back to regular scatter plot")
            return self._create_3d_scatter_plot(
                df, x_column, y_column, z_column, color_by, size_by, text_column,
                output_path, title, viz_3d_config
            )
    
    def _add_regression_plane(self, fig, df, x_column, y_column, z_column, viz_3d_config):
        """
        Add a regression plane to a 3D scatter plot.
        
        Args:
            fig: Plotly figure to add the plane to
            df: DataFrame with data points
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            viz_3d_config: Configuration for the visualization
        """
        try:
            # Create data for multiple linear regression
            X = df[[x_column, y_column]].values
            y = df[z_column].values
            
            # Add constant term for intercept
            X_with_const = np.column_stack((np.ones(X.shape[0]), X))
            
            # Solve for coefficients using least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            
            # Extract coefficients
            intercept, a, b = coeffs
            
            # Create meshgrid for plane
            x_range = np.linspace(df[x_column].min(), df[x_column].max(), 20)
            y_range = np.linspace(df[y_column].min(), df[y_column].max(), 20)
            x_mesh, y_mesh = np.meshgrid(x_range, y_range)
            
            # Calculate z values for the plane
            z_mesh = intercept + a * x_mesh + b * y_mesh
            
            # Add regression plane
            fig.add_trace(go.Surface(
                x=x_range,
                y=y_range,
                z=z_mesh,
                opacity=viz_3d_config.get("surface_opacity", 0.7) * 0.7,
                colorscale=[[0, "rgba(200,200,200,0.5)"], [1, "rgba(200,200,200,0.5)"]],
                showscale=False,
                name='Regression Plane',
                hoverinfo='skip'
            ))
            
            # Calculate R-squared value
            y_pred = intercept + a * df[x_column].values + b * df[y_column].values
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Add annotation with regression equation and R-squared
            equation = f"z = {intercept:.3f} + {a:.3f}x + {b:.3f}y"
            r2_text = f"R² = {r_squared:.3f}"
            
            fig.add_annotation(
                x=0.05,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"{equation}<br>{r2_text}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)" if self.theme == "light" else "rgba(0,0,0,0.8)",
                bordercolor=self.theme_colors.get("accent1", "blue"),
                borderwidth=1,
                borderpad=4
            )
            
        except Exception as e:
            logger.warning(f"Error adding regression plane: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    def _add_3d_projections(self, fig, df, x_column, y_column, z_column, viz_3d_config):
        """
        Add projections to the walls of a 3D scatter plot.
        
        Args:
            fig: Plotly figure to add projections to
            df: DataFrame with data points
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            viz_3d_config: Configuration for the visualization
        """
        try:
            # Add x-z projection (to y=min wall)
            fig.add_trace(go.Scatter3d(
                x=df[x_column],
                y=np.ones_like(df[x_column]) * df[y_column].min(),
                z=df[z_column],
                mode='markers',
                marker=dict(
                    size=viz_3d_config["marker_size"] * 0.7,
                    color=self.theme_colors.get("accent1", "#1f77b4"),
                    opacity=0.3
                ),
                name='X-Z Projection',
                showlegend=False
            ))
            
            # Add y-z projection (to x=min wall)
            fig.add_trace(go.Scatter3d(
                x=np.ones_like(df[y_column]) * df[x_column].min(),
                y=df[y_column],
                z=df[z_column],
                mode='markers',
                marker=dict(
                    size=viz_3d_config["marker_size"] * 0.7,
                    color=self.theme_colors.get("accent2", "#ff7f0e"),
                    opacity=0.3
                ),
                name='Y-Z Projection',
                showlegend=False
            ))
            
            # Add x-y projection (to z=min wall)
            fig.add_trace(go.Scatter3d(
                x=df[x_column],
                y=df[y_column],
                z=np.ones_like(df[x_column]) * df[z_column].min(),
                mode='markers',
                marker=dict(
                    size=viz_3d_config["marker_size"] * 0.7,
                    color=self.theme_colors.get("accent3", "#2ca02c"),
                    opacity=0.3
                ),
                name='X-Y Projection',
                showlegend=False
            ))
            
        except Exception as e:
            logger.warning(f"Error adding 3D projections: {e}")
    
    def _create_static_3d(self, df, x_column, y_column, z_column,
                         color_by, size_by, text_column,
                         output_path, title, viz_3d_config):
        """
        Create a static 3D visualization using Matplotlib.
        
        Args:
            df: DataFrame with performance data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            z_column: Column name for z-axis
            color_by: Column to use for coloring points
            size_by: Optional column to use for sizing points
            text_column: Optional column to use for point labels
            output_path: Path to save the visualization
            title: Title for the visualization
            viz_3d_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Create figure
            fig = plt.figure(figsize=(viz_3d_config["width"] / 100, viz_3d_config["height"] / 100))
            ax = fig.add_subplot(111, projection='3d')
            
            # Determine marker size
            marker_size = viz_3d_config["marker_size"]
            if size_by and size_by in df.columns:
                # Normalize size column to a reasonable range
                size_min = df[size_by].min()
                size_max = df[size_by].max()
                if size_min != size_max:  # Avoid division by zero
                    size_norm = (df[size_by] - size_min) / (size_max - size_min)
                    # Scale to desired size range
                    marker_size = 20 + size_norm * 100
                else:
                    marker_size = 50
            
            # Determine coloring
            if color_by and color_by in df.columns:
                unique_values = df[color_by].nunique()
                
                # If few unique values or string type, treat as categorical
                if unique_values <= 10 or pd.api.types.is_string_dtype(df[color_by]):
                    # Create a scatter for each category
                    categories = df[color_by].unique()
                    
                    # Get color map
                    cmap = plt.cm.get_cmap('tab10', len(categories))
                    
                    # Create a scatter for each category
                    for i, category in enumerate(sorted(categories)):
                        category_df = df[df[color_by] == category]
                        
                        # Determine marker size for this category
                        if size_by and size_by in df.columns:
                            category_marker_size = marker_size[df[color_by] == category]
                        else:
                            category_marker_size = marker_size
                        
                        ax.scatter(
                            category_df[x_column],
                            category_df[y_column],
                            category_df[z_column],
                            s=category_marker_size,
                            c=[cmap(i)],
                            label=str(category),
                            alpha=viz_3d_config["marker_opacity"]
                        )
                else:
                    # Continuous coloring
                    scatter = ax.scatter(
                        df[x_column],
                        df[y_column],
                        df[z_column],
                        s=marker_size,
                        c=df[color_by],
                        cmap=viz_3d_config.get("colormap", "viridis"),
                        alpha=viz_3d_config["marker_opacity"]
                    )
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
                    cbar.set_label(self._get_metric_label(color_by))
            else:
                # Simple single color 3D scatter
                ax.scatter(
                    df[x_column],
                    df[y_column],
                    df[z_column],
                    s=marker_size,
                    c=self.theme_colors.get("accent1", "#1f77b4"),
                    alpha=viz_3d_config["marker_opacity"]
                )
            
            # Add regression plane if requested
            if viz_3d_config.get("regression_plane") and len(df) >= 10:
                try:
                    # Create data for multiple linear regression
                    X = df[[x_column, y_column]].values
                    y = df[z_column].values
                    
                    # Add constant term for intercept
                    X_with_const = np.column_stack((np.ones(X.shape[0]), X))
                    
                    # Solve for coefficients using least squares
                    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
                    
                    # Extract coefficients
                    intercept, a, b = coeffs
                    
                    # Create meshgrid for plane
                    x_range = np.linspace(df[x_column].min(), df[x_column].max(), 20)
                    y_range = np.linspace(df[y_column].min(), df[y_column].max(), 20)
                    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
                    
                    # Calculate z values for the plane
                    z_mesh = intercept + a * x_mesh + b * y_mesh
                    
                    # Add regression plane
                    ax.plot_surface(
                        x_mesh, y_mesh, z_mesh,
                        alpha=viz_3d_config.get("surface_opacity", 0.7) * 0.5,
                        color='gray',
                        label='Regression Plane'
                    )
                    
                    # Calculate R-squared value
                    y_pred = intercept + a * df[x_column].values + b * df[y_column].values
                    ss_total = np.sum((y - np.mean(y)) ** 2)
                    ss_residual = np.sum((y - y_pred) ** 2)
                    r_squared = 1 - (ss_residual / ss_total)
                    
                    # Add text annotation with regression equation and R-squared
                    equation = f"z = {intercept:.3f} + {a:.3f}x + {b:.3f}y"
                    r2_text = f"R² = {r_squared:.3f}"
                    
                    # Add text to the figure
                    ax.text2D(
                        0.05, 0.95, 
                        equation + "\n" + r2_text,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white" if self.theme == "light" else "black",
                            alpha=0.8
                        )
                    )
                    
                except Exception as e:
                    logger.warning(f"Error adding regression plane: {e}")
            
            # Add annotations if requested
            if viz_3d_config.get("show_annotations") and text_column and text_column in df.columns:
                for i, row in df.iterrows():
                    ax.text(
                        row[x_column], row[y_column], row[z_column],
                        str(row[text_column]),
                        size=viz_3d_config["annotation_size"],
                        zorder=1
                    )
            
            # Set labels and title
            ax.set_xlabel(self._get_metric_label(x_column))
            ax.set_ylabel(self._get_metric_label(y_column))
            ax.set_zlabel(self._get_metric_label(z_column))
            ax.set_title(title)
            
            # Add grid
            if viz_3d_config.get("axis_grid", True):
                ax.grid(True)
            
            # Add legend if we have categorical colors
            if color_by and df[color_by].nunique() <= 10:
                ax.legend()
            
            # Set theme-specific colors
            if self.theme == "dark":
                plt.style.use('dark_background')
                fig.patch.set_facecolor('#333333')
                ax.set_facecolor('#333333')
            
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
            logger.error(f"Error creating static 3D visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None