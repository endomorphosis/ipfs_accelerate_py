#!/usr/bin/env python3
"""
Advanced Visualization System for IPFS Accelerate

This module implements interactive 3D visualizations for multi-dimensional performance data.
It builds on the existing visualization engine to provide more advanced and interactive
visualization capabilities.

Features:
- Interactive 3D plots for performance data visualization
- Dynamic hardware comparison heatmaps by model families
- Power efficiency visualization with interactive filters
- Animated time-series performance visualizations
- Customizable dashboard system with user configuration
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
logger = logging.getLogger("advanced_visualization")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Install with: pip install pandas")
    PANDAS_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn not available. Some dimensionality reduction features will be limited.")
    SKLEARN_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    logger.warning("IPython widgets not available. Interactive filtering features will be limited.")
    IPYWIDGETS_AVAILABLE = False

# Import dashboard integration
try:
    from duckdb_api.visualization.advanced_visualization.monitor_dashboard_integration import (
        MonitorDashboardIntegrationMixin
    )
    HAS_DASHBOARD_INTEGRATION = True
except ImportError:
    HAS_DASHBOARD_INTEGRATION = False
    # Create a dummy mixin if not available
    class MonitorDashboardIntegrationMixin:
        def __init__(self, dashboard_url=None, api_key=None):
            pass

# Set base classes for AdvancedVisualizationSystem
if HAS_DASHBOARD_INTEGRATION:
    AdvancedVisualizationBaseClasses = (MonitorDashboardIntegrationMixin,)
else:
    AdvancedVisualizationBaseClasses = ()

class AdvancedVisualizationSystem(*AdvancedVisualizationBaseClasses):
    """Advanced Visualization System for multi-dimensional performance data.
    
    This class provides comprehensive visualization capabilities for analyzing model performance
    across different hardware platforms, batch sizes, and precision formats."""
    
    def __init__(self, db_api=None, output_dir: str = "./advanced_visualizations", dashboard_url=None, api_key=None):
        """Initialize the advanced visualization system.
        
        Args:
            db_api: Database API for accessing performance data
            output_dir: Directory to save visualizations
            dashboard_url: URL of the monitoring dashboard (default: None)
            api_key: API key for dashboard authentication (default: None)
        """
        # Initialize dashboard integration mixin if available
        if HAS_DASHBOARD_INTEGRATION and issubclass(self.__class__, MonitorDashboardIntegrationMixin):
            MonitorDashboardIntegrationMixin.__init__(self, dashboard_url=dashboard_url, api_key=api_key)
        self.db_api = db_api
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dashboards directory
        self.dashboards_dir = os.path.join(output_dir, "dashboards")
        os.makedirs(self.dashboards_dir, exist_ok=True)
        
        # Default configuration
        self.config = {
            "theme": "light",           # light or dark
            "color_palette": "viridis",  # viridis, plasma, inferno, etc.
            "default_width": 1000,      # Default width in pixels
            "default_height": 800,      # Default height in pixels
            "auto_open": True,          # Automatically open visualizations in browser
            "include_annotations": True, # Include annotations on charts
            "animation_duration": 1000,  # Animation duration in milliseconds
            "include_controls": True,    # Include interactive controls
            "save_data": True,          # Save data alongside visualizations
            "dashboard_columns": 2,     # Default number of columns in dashboard grid
            "dashboard_row_height": 600, # Default height for dashboard rows
            "dashboard_padding": 20,    # Padding between dashboard components
        }
        
        # Dashboard templates and configurations
        self.dashboard_templates = {
            "overview": {
                "title": "Performance Overview Dashboard",
                "description": "Overview of performance metrics across models and hardware",
                "components": [
                    {"type": "3d", "metrics": ["throughput", "latency", "memory_usage"], "width": 1, "height": 1},
                    {"type": "heatmap", "metric": "throughput", "width": 1, "height": 1},
                    {"type": "time-series", "metric": "throughput", "width": 2, "height": 1},
                ]
            },
            "hardware_comparison": {
                "title": "Hardware Comparison Dashboard",
                "description": "Detailed comparison of performance across hardware platforms",
                "components": [
                    {"type": "heatmap", "metric": "throughput", "width": 2, "height": 1},
                    {"type": "power", "width": 1, "height": 1},
                    {"type": "time-series", "metric": "throughput", "dimensions": ["hardware_type"], "width": 1, "height": 1},
                ]
            },
            "model_analysis": {
                "title": "Model Analysis Dashboard",
                "description": "Detailed analysis of model performance metrics",
                "components": [
                    {"type": "3d", "metrics": ["throughput", "latency", "memory_usage"], "width": 1, "height": 1},
                    {"type": "time-series", "metric": "throughput", "dimensions": ["model_family"], "width": 1, "height": 1},
                    {"type": "heatmap", "metric": "latency", "width": 2, "height": 1},
                ]
            },
            "empty": {
                "title": "Custom Dashboard",
                "description": "A blank dashboard for custom visualizations",
                "components": []
            }
        }
        
        # Store for saved dashboards
        self.saved_dashboards = {}
        self.dashboard_config_path = os.path.join(self.dashboards_dir, "dashboard_configs.json")
        
        # Try to load saved dashboard configurations if they exist
        if os.path.exists(self.dashboard_config_path):
            try:
                with open(self.dashboard_config_path, 'r') as f:
                    self.saved_dashboards = json.load(f)
                logger.info(f"Loaded {len(self.saved_dashboards)} saved dashboard configurations")
            except Exception as e:
                logger.error(f"Error loading saved dashboards: {e}")
        
        # Color schemes
        self.color_schemes = {
            "light": {
                "bg_color": "white",
                "text_color": "black",
                "grid_color": "#eee",
                "accent_color": "#1f77b4",
                "highlight_color": "#ff7f0e",
                "positive_color": "#2ca02c",
                "negative_color": "#d62728",
                "neutral_color": "#7f7f7f",
                "dashboard_bg": "#f9f9f9",
                "card_bg": "white",
                "card_border": "#ddd",
                "header_bg": "#f5f5f5",
            },
            "dark": {
                "bg_color": "#111111",
                "text_color": "#ffffff",
                "grid_color": "#333333",
                "accent_color": "#1f77b4",
                "highlight_color": "#ff7f0e",
                "positive_color": "#2ca02c",
                "negative_color": "#d62728",
                "neutral_color": "#7f7f7f",
                "dashboard_bg": "#1a1a1a",
                "card_bg": "#2d2d2d",
                "card_border": "#444",
                "header_bg": "#222",
            }
        }
        
        # Current color scheme based on theme
        self.colors = self.color_schemes[self.config["theme"]]
        
        logger.info("Advanced Visualization System initialized")
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update the visualization system configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        
        # Update colors based on theme if changed
        if "theme" in config_updates:
            self.colors = self.color_schemes[self.config["theme"]]
        
        logger.info(f"Configuration updated: {config_updates}")
    
    def create_3d_performance_visualization(self,
                                          data: Optional[Dict[str, Any]] = None,
                                          metrics: List[str] = ["throughput", "latency", "memory_usage"],
                                          dimensions: List[str] = ["model", "hardware", "batch_size"],
                                          filters: Optional[Dict[str, List[str]]] = None,
                                          output_path: Optional[str] = None,
                                          title: str = "3D Performance Visualization") -> Optional[str]:
        """Create an interactive 3D visualization of performance data.
        
        Args:
            data: Optional performance data dictionary (will use db_api if None)
            metrics: List of metrics to include in visualization (min 3 needed for 3D)
            dimensions: List of dimensions to use for grouping and coloring
            filters: Optional filters to apply to the data
            output_path: Optional path for saving the visualization (auto-generated if None)
            title: Title for the visualization
            
        Returns:
            Path to the saved visualization HTML file, or None if error
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.error("Plotly and Pandas are required for 3D visualization.")
            return None
        
        if len(metrics) < 3:
            logger.error("At least 3 metrics are required for 3D visualization.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"3d_performance_{timestamp}.html")
        
        try:
            # Get data if not provided
            if data is None and self.db_api:
                data = self._get_performance_data(metrics, dimensions, filters)
            
            if not data or "data_points" not in data:
                logger.error("No data available for visualization.")
                return None
            
            # Convert data to DataFrame
            df = pd.DataFrame(data["data_points"])
            
            if len(df) == 0:
                logger.error("No data points available for visualization.")
                return None
            
            # Extract main metrics for 3D coordinates
            x_metric = metrics[0]
            y_metric = metrics[1]
            z_metric = metrics[2]
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            # Determine coloring dimension
            color_dim = dimensions[0] if dimensions else None
            
            # For each unique value in the coloring dimension, add a separate trace
            if color_dim and color_dim in df.columns:
                for value in df[color_dim].unique():
                    subset = df[df[color_dim] == value]
                    
                    fig.add_trace(go.Scatter3d(
                        x=subset[x_metric],
                        y=subset[y_metric],
                        z=subset[z_metric],
                        mode='markers',
                        marker=dict(
                            size=8,
                            opacity=0.8,
                        ),
                        name=f"{color_dim}={value}",
                        text=[self._create_hover_text(row, metrics, dimensions) for _, row in subset.iterrows()],
                        hoverinfo='text'
                    ))
            else:
                # Create a single trace with all data points
                fig.add_trace(go.Scatter3d(
                    x=df[x_metric],
                    y=df[y_metric],
                    z=df[z_metric],
                    mode='markers',
                    marker=dict(
                        size=8,
                        opacity=0.8,
                        color=df.index,  # Color by index if no dimension specified
                        colorscale=self.config["color_palette"],
                    ),
                    text=[self._create_hover_text(row, metrics, dimensions) for _, row in df.iterrows()],
                    hoverinfo='text'
                ))
            
            # Layout configuration
            layout = dict(
                title=title,
                scene=dict(
                    xaxis=dict(title=x_metric.replace('_', ' ').title()),
                    yaxis=dict(title=y_metric.replace('_', ' ').title()),
                    zaxis=dict(title=z_metric.replace('_', ' ').title()),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                template="plotly_white" if self.config["theme"] == "light" else "plotly_dark",
            )
            
            # Update layout
            fig.update_layout(layout)
            
            # Add interactive controls if requested
            if self.config["include_controls"]:
                # Add buttons for different camera angles
                camera_buttons = [
                    dict(
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.2}}],
                        label="Default View"
                    ),
                    dict(
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}],
                        label="Top View"
                    ),
                    dict(
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0}}],
                        label="Side View"
                    ),
                    dict(
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 0, "y": 2.5, "z": 0}}],
                        label="Front View"
                    ),
                ]
                
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            buttons=camera_buttons,
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0,
                            xanchor="left",
                            y=0,
                            yanchor="top",
                            bgcolor=self.colors["bg_color"],
                            bordercolor=self.colors["accent_color"],
                        )
                    ]
                )
            
            # Save visualization to HTML file
            pio.write_html(
                fig,
                file=output_path,
                auto_open=self.config["auto_open"],
                include_plotlyjs='cdn',
            )
            
            # Optionally save data for later use
            if self.config["save_data"]:
                data_path = output_path.replace('.html', '_data.json')
                with open(data_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"3D visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def create_hardware_comparison_heatmap(self,
                                         data: Optional[Dict[str, Any]] = None,
                                         metric: str = "throughput",
                                         model_families: Optional[List[str]] = None,
                                         hardware_types: Optional[List[str]] = None,
                                         batch_size: int = 1,
                                         output_path: Optional[str] = None,
                                         title: str = "Hardware Comparison by Model Family") -> Optional[str]:
        """Create a dynamic hardware comparison heatmap grouped by model families.
        
        Args:
            data: Optional performance data dictionary (will use db_api if None)
            metric: Metric to visualize
            model_families: Optional list of model families to include
            hardware_types: Optional list of hardware types to include
            batch_size: Batch size to use for comparison
            output_path: Optional path for saving the visualization
            title: Title for the visualization
            
        Returns:
            Path to the saved visualization HTML file, or None if error
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.error("Plotly and Pandas are required for hardware comparison heatmap.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"hardware_model_heatmap_{timestamp}.html")
        
        try:
            # Get data if not provided
            if data is None and self.db_api:
                # Build query to get model family and hardware data
                query = """
                WITH latest_results AS (
                    SELECT 
                        m.model_id,
                        m.model_name,
                        m.model_family,
                        hp.hardware_id,
                        hp.hardware_type,
                        hp.hardware_specs,
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
                
                if model_families:
                    model_list = ", ".join([f"'{model}'" for model in model_families])
                    query += f" AND m.model_family IN ({model_list})"
                
                if hardware_types:
                    hw_list = ", ".join([f"'{hw}'" for hw in hardware_types])
                    query += f" AND hp.hardware_type IN ({hw_list})"
                
                query += """
                )
                SELECT
                    model_id,
                    model_name,
                    model_family,
                    hardware_id,
                    hardware_type,
                    hardware_specs,
                    batch_size,
                    average_latency_ms,
                    throughput_items_per_second,
                    memory_peak_mb,
                    is_simulated
                FROM
                    latest_results
                WHERE
                    rn = 1
                ORDER BY
                    model_family, model_name, hardware_type
                """
                
                # Execute query
                results = self.db_api.query(query, params)
                
                if results.empty:
                    logger.error("No data found for the specified parameters.")
                    return None
                
                # Convert results to the required format
                data = {
                    "model_families": results["model_family"].unique().tolist(),
                    "hardware_types": results["hardware_type"].unique().tolist(),
                    "data_points": results.to_dict(orient="records")
                }
            
            if not data:
                logger.error("No data available for visualization.")
                return None
            
            # Convert data to DataFrame
            df = pd.DataFrame(data["data_points"])
            
            if len(df) == 0:
                logger.error("No data points available for visualization.")
                return None
            
            # Prepare data for heatmap
            # Pivot the data for each model family
            model_families = data.get("model_families", df["model_family"].unique().tolist())
            hardware_types = data.get("hardware_types", df["hardware_type"].unique().tolist())
            
            if metric == "throughput":
                metric_col = "throughput_items_per_second"
                metric_title = "Throughput (items/sec)"
                # Higher is better for throughput
                color_scale = self.config["color_palette"]
            elif metric == "latency":
                metric_col = "average_latency_ms"
                metric_title = "Latency (ms)"
                # Lower is better for latency (reverse colorscale)
                color_scale = self.config["color_palette"] + "_r"
            elif metric == "memory":
                metric_col = "memory_peak_mb"
                metric_title = "Memory Usage (MB)"
                # Lower is better for memory (reverse colorscale)
                color_scale = self.config["color_palette"] + "_r"
            else:
                logger.error(f"Unknown metric: {metric}")
                return None
            
            # Create a subplot for each model family
            fig = make_subplots(
                rows=len(model_families),
                cols=1,
                subplot_titles=[f"{family} Models" for family in model_families],
                vertical_spacing=0.05
            )
            
            # Add annotations to show simulated results
            simulated_annotations = []
            
            # For each model family, create a heatmap
            for i, family in enumerate(model_families):
                family_df = df[df["model_family"] == family]
                
                # If no data for this family, skip
                if len(family_df) == 0:
                    continue
                
                # Get unique models in this family
                models = family_df["model_name"].unique()
                
                # Create a new dataframe with all hardware types and models
                heatmap_data = np.full((len(models), len(hardware_types)), np.nan)
                
                # Fill in available data
                for j, model in enumerate(models):
                    model_df = family_df[family_df["model_name"] == model]
                    for k, hw in enumerate(hardware_types):
                        hw_df = model_df[model_df["hardware_type"] == hw]
                        if len(hw_df) > 0:
                            heatmap_data[j, k] = hw_df[metric_col].values[0]
                            
                            # Mark simulated results
                            if hw_df["is_simulated"].values[0]:
                                simulated_annotations.append(
                                    dict(
                                        x=k,
                                        y=j,
                                        xref=f"x{i+1}" if i > 0 else "x",
                                        yref=f"y{i+1}" if i > 0 else "y",
                                        text="S",
                                        showarrow=False,
                                        font=dict(color="white", size=10),
                                        bgcolor="rgba(0,0,0,0.5)",
                                        bordercolor="white",
                                        borderwidth=1,
                                        borderpad=1,
                                        opacity=0.8
                                    )
                                )
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data,
                        x=hardware_types,
                        y=models,
                        colorscale=color_scale,
                        colorbar=dict(
                            title=metric_title,
                            len=1/len(model_families),
                            y=(1-(2*i+1)/(2*len(model_families)))
                        ),
                        hovertemplate=(
                            "<b>Model:</b> %{y}<br>"
                            "<b>Hardware:</b> %{x}<br>"
                            f"<b>{metric_title}:</b> %{{z:.2f}}<br>"
                        ),
                        name=family
                    ),
                    row=i+1, col=1
                )
            
            # Update layout
            height_per_family = 300  # Height per model family subplot
            fig.update_layout(
                title=title,
                height=height_per_family * len(model_families),
                width=self.config["default_width"],
                template="plotly_white" if self.config["theme"] == "light" else "plotly_dark",
                annotations=simulated_annotations,
            )
            
            # Update y-axis properties
            for i in range(1, len(model_families)+1):
                fig.update_yaxes(
                    title="Model",
                    autorange="reversed",
                    row=i, col=1
                )
            
            # Only add x-axis title to the bottom subplot
            fig.update_xaxes(
                title="Hardware Type",
                row=len(model_families), col=1
            )
            
            # Add legend for simulated results
            fig.add_annotation(
                x=1.05,
                y=1,
                xref="paper",
                yref="paper",
                text="S = Simulated Result",
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="rgba(255,255,255,0.8)" if self.config["theme"] == "light" else "rgba(0,0,0,0.8)",
                bordercolor="black" if self.config["theme"] == "light" else "white",
                borderwidth=1,
                borderpad=4
            )
            
            # Add interactive controls if requested
            if self.config["include_controls"]:
                # Add buttons to toggle simulated results
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="down",
                            buttons=[
                                dict(
                                    args=[{"annotations": simulated_annotations}],
                                    label="Show Simulated Markers",
                                    method="relayout"
                                ),
                                dict(
                                    args=[{"annotations": []}],
                                    label="Hide Simulated Markers",
                                    method="relayout"
                                ),
                            ],
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=1.05,
                            xanchor="left",
                            y=0.5,
                            yanchor="middle",
                            bgcolor=self.colors["bg_color"],
                            bordercolor=self.colors["accent_color"],
                        )
                    ]
                )
            
            # Save visualization to HTML file
            pio.write_html(
                fig,
                file=output_path,
                auto_open=self.config["auto_open"],
                include_plotlyjs='cdn',
            )
            
            # Optionally save data for later use
            if self.config["save_data"]:
                data_path = output_path.replace('.html', '_data.json')
                with open(data_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Hardware comparison heatmap saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating hardware comparison heatmap: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_performance_data(self, metrics, dimensions, filters):
        """Get performance data from the database.
        
        Args:
            metrics: List of metrics to include
            dimensions: List of dimensions to include
            filters: Optional filters to apply
            
        Returns:
            Dictionary containing performance data
        """
        if not self.db_api:
            logger.error("No database API available.")
            return None
        
        # Build the base query
        query = """
        WITH latest_results AS (
            SELECT 
                m.model_id,
                m.model_name,
                m.model_family,
                hp.hardware_id,
                hp.hardware_type,
                hp.hardware_specs,
                hp.hardware_vendor,
                pr.batch_size,
                pr.precision,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                pr.energy_consumption_joules,
                pr.is_simulated,
                ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size, pr.precision
                ORDER BY pr.created_at DESC) as rn
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        """
        
        # Add filters if specified
        where_clauses = []
        params = {}
        
        if filters:
            for key, values in filters.items():
                if values:
                    placeholder = f"{key}_values"
                    where_clauses.append(f"{key} IN (:{placeholder})")
                    params[placeholder] = values
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += """
        )
        SELECT
            model_id,
            model_name,
            model_family,
            hardware_id,
            hardware_type,
            hardware_vendor,
            batch_size,
            precision,
            average_latency_ms,
            throughput_items_per_second,
            memory_peak_mb,
            energy_consumption_joules,
            is_simulated
        FROM
            latest_results
        WHERE
            rn = 1
        """
        
        # Execute query
        results = self.db_api.query(query, params)
        
        if results.empty:
            logger.error("No data found for the specified parameters.")
            return None
        
        # Convert results to the required format
        return {
            "metrics": metrics,
            "dimensions": dimensions,
            "data_points": results.to_dict(orient="records")
        }
    
    def _create_hover_text(self, row, metrics, dimensions):
        """Create hover text for 3D scatter plot points.
        
        Args:
            row: DataFrame row
            metrics: List of metrics
            dimensions: List of dimensions
            
        Returns:
            Formatted hover text
        """
        hover_text = []
        
        # Add dimensions
        for dim in dimensions:
            if dim in row:
                hover_text.append(f"<b>{dim}:</b> {row[dim]}")
        
        # Add metrics
        for metric in metrics:
            if metric in row:
                value = row[metric]
                hover_text.append(f"<b>{metric}:</b> {value:.2f}")
        
        # Add simulation status if available
        if "is_simulated" in row:
            hover_text.append(f"<b>Simulated:</b> {'Yes' if row['is_simulated'] else 'No'}")
        
        return "<br>".join(hover_text)

    def create_power_efficiency_visualization(self,
                                           data: Optional[Dict[str, Any]] = None,
                                           hardware_types: Optional[List[str]] = None,
                                           model_families: Optional[List[str]] = None,
                                           batch_sizes: Optional[List[int]] = None,
                                           output_path: Optional[str] = None,
                                           title: str = "Power Efficiency Visualization") -> Optional[str]:
        """Create an interactive power efficiency visualization with filters.
        
        This visualization shows the relationship between throughput and energy consumption,
        with points sized by latency and colored by efficiency (throughput per joule).
        
        Args:
            data: Optional performance data dictionary (will use db_api if None)
            hardware_types: Optional list of hardware types to include
            model_families: Optional list of model families to include
            batch_sizes: Optional list of batch sizes to include
            output_path: Optional path for saving the visualization
            title: Title for the visualization
            
        Returns:
            Path to the saved visualization HTML file, or None if error
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.error("Plotly and Pandas are required for power efficiency visualization.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"power_efficiency_{timestamp}.html")
        
        try:
            # Get data if not provided
            if data is None and self.db_api:
                # Build filters
                filters = {}
                if hardware_types:
                    filters["hardware_type"] = hardware_types
                if model_families:
                    filters["model_family"] = model_families
                if batch_sizes:
                    filters["batch_size"] = batch_sizes
                
                # Get data
                metrics = ["throughput_items_per_second", "energy_consumption_joules", "average_latency_ms"]
                dimensions = ["hardware_type", "model_family", "batch_size", "precision"]
                data = self._get_performance_data(metrics, dimensions, filters)
            
            if not data or "data_points" not in data:
                logger.error("No data available for visualization.")
                return None
            
            # Convert data to DataFrame
            df = pd.DataFrame(data["data_points"])
            
            if len(df) == 0:
                logger.error("No data points available for visualization.")
                return None
            
            # Calculate power efficiency (throughput per joule)
            # Higher values mean better efficiency
            if "throughput_items_per_second" in df.columns and "energy_consumption_joules" in df.columns:
                df["efficiency"] = df["throughput_items_per_second"] / df["energy_consumption_joules"]
            else:
                logger.error("Required metrics not available in data.")
                return None
            
            # Create interactive figure
            fig = go.Figure()
            
            # Add scatter plot of throughput vs. energy consumption, with size representing latency
            fig.add_trace(go.Scatter(
                x=df["energy_consumption_joules"],
                y=df["throughput_items_per_second"],
                mode="markers",
                marker=dict(
                    size=df["average_latency_ms"].apply(lambda x: max(10, min(50, 200 / (x + 1)))),  # Scale marker size
                    color=df["efficiency"],
                    colorscale=self.config["color_palette"],
                    showscale=True,
                    colorbar=dict(
                        title="Efficiency<br>(Throughput/Joule)",
                        thickness=20,
                        len=0.7,
                    ),
                    opacity=0.8,
                ),
                text=[
                    f"<b>Model Family:</b> {row['model_family']}<br>" +
                    f"<b>Hardware:</b> {row['hardware_type']}<br>" +
                    f"<b>Batch Size:</b> {row['batch_size']}<br>" +
                    f"<b>Precision:</b> {row['precision']}<br>" +
                    f"<b>Throughput:</b> {row['throughput_items_per_second']:.2f} items/s<br>" +
                    f"<b>Energy:</b> {row['energy_consumption_joules']:.2f} joules<br>" +
                    f"<b>Latency:</b> {row['average_latency_ms']:.2f} ms<br>" +
                    f"<b>Efficiency:</b> {row['efficiency']:.2f} items/joule"
                    for _, row in df.iterrows()
                ],
                hoverinfo="text",
                name="Efficiency"
            ))
            
            # Create a line representing constant efficiency values
            if len(df) > 0:
                # Calculate a range of energy values
                max_energy = df["energy_consumption_joules"].max()
                energy_values = np.linspace(0, max_energy * 1.2, 100)
                
                # Add efficiency lines
                efficiency_levels = [0.1, 0.5, 1, 5, 10, 50, 100]
                for efficiency in efficiency_levels:
                    # Skip efficiency levels that are too high or too low
                    if (efficiency * max_energy > df["throughput_items_per_second"].max() * 3 or 
                        efficiency * max_energy < df["throughput_items_per_second"].min() / 3):
                        continue
                    
                    fig.add_trace(go.Scatter(
                        x=energy_values,
                        y=energy_values * efficiency,
                        mode="lines",
                        line=dict(
                            color="rgba(0, 0, 0, 0.3)",
                            width=1,
                            dash="dash"
                        ),
                        name=f"Efficiency = {efficiency}",
                        hoverinfo="name"
                    ))
            
            # Add reference lines for axes
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=df["energy_consumption_joules"].max() * 1.2,
                y1=0,
                line=dict(color="rgba(0,0,0,0.3)", width=1)
            )
            
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=0,
                y1=df["throughput_items_per_second"].max() * 1.2,
                line=dict(color="rgba(0,0,0,0.3)", width=1)
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis=dict(
                    title="Energy Consumption (joules)",
                    gridcolor=self.colors["grid_color"],
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Throughput (items per second)",
                    gridcolor=self.colors["grid_color"],
                    zeroline=False,
                ),
                width=self.config["default_width"],
                height=self.config["default_height"],
                template="plotly_white" if self.config["theme"] == "light" else "plotly_dark",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=30, t=50, b=60),
            )
            
            # Add filters using sliders and dropdowns
            if self.config["include_controls"]:
                # Create filter controls
                sliders = []
                dropdowns = []
                
                # Model family dropdown
                if "model_family" in df.columns and len(df["model_family"].unique()) > 1:
                    model_family_buttons = [
                        dict(
                            args=[{
                                "visible": [True] * len(fig.data)
                            }],
                            label="All Model Families",
                            method="restyle"
                        )
                    ]
                    
                    for family in sorted(df["model_family"].unique()):
                        # Create a visibility list: only show points from this family
                        # and always show efficiency lines
                        visibility = []
                        for trace_idx, trace in enumerate(fig.data):
                            if trace_idx == 0:  # First trace is the scatter plot
                                visibility.append(False)
                            else:  # Other traces are efficiency lines
                                visibility.append(True)
                        
                        # Create a filtered scatter plot for this family
                        family_df = df[df["model_family"] == family]
                        fig.add_trace(go.Scatter(
                            x=family_df["energy_consumption_joules"],
                            y=family_df["throughput_items_per_second"],
                            mode="markers",
                            marker=dict(
                                size=family_df["average_latency_ms"].apply(lambda x: max(10, min(50, 200 / (x + 1)))),
                                color=family_df["efficiency"],
                                colorscale=self.config["color_palette"],
                                showscale=False,
                                opacity=0.8,
                            ),
                            text=[
                                f"<b>Model Family:</b> {row['model_family']}<br>" +
                                f"<b>Hardware:</b> {row['hardware_type']}<br>" +
                                f"<b>Batch Size:</b> {row['batch_size']}<br>" +
                                f"<b>Precision:</b> {row['precision']}<br>" +
                                f"<b>Throughput:</b> {row['throughput_items_per_second']:.2f} items/s<br>" +
                                f"<b>Energy:</b> {row['energy_consumption_joules']:.2f} joules<br>" +
                                f"<b>Latency:</b> {row['average_latency_ms']:.2f} ms<br>" +
                                f"<b>Efficiency:</b> {row['efficiency']:.2f} items/joule"
                                for _, row in family_df.iterrows()
                            ],
                            hoverinfo="text",
                            name=f"{family}",
                            visible=False  # Initially hidden
                        ))
                        
                        # Add a button to show only this family
                        button_visibility = [False] + [True] * (len(fig.data) - 2) + [False] * (len(fig.data) - len(visibility) - 1)
                        button_visibility[-1] = True  # Show the new trace
                        
                        model_family_buttons.append(
                            dict(
                                args=[{"visible": button_visibility}],
                                label=family,
                                method="restyle"
                            )
                        )
                    
                    # Add dropdown menu
                    dropdowns.append(
                        dict(
                            buttons=model_family_buttons,
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.1,
                            yanchor="top",
                            bgcolor=self.colors["bg_color"],
                            bordercolor=self.colors["accent_color"],
                            type="dropdown",
                            font=dict(color=self.colors["text_color"]),
                        )
                    )
                
                # Hardware type dropdown
                if "hardware_type" in df.columns and len(df["hardware_type"].unique()) > 1:
                    hardware_buttons = [
                        dict(
                            args=[{
                                "visible": [True] * len(fig.data)
                            }],
                            label="All Hardware Types",
                            method="restyle"
                        )
                    ]
                    
                    for hw_type in sorted(df["hardware_type"].unique()):
                        # Create a filtered scatter plot for this hardware type
                        hw_df = df[df["hardware_type"] == hw_type]
                        fig.add_trace(go.Scatter(
                            x=hw_df["energy_consumption_joules"],
                            y=hw_df["throughput_items_per_second"],
                            mode="markers",
                            marker=dict(
                                size=hw_df["average_latency_ms"].apply(lambda x: max(10, min(50, 200 / (x + 1)))),
                                color=hw_df["efficiency"],
                                colorscale=self.config["color_palette"],
                                showscale=False,
                                opacity=0.8,
                            ),
                            text=[
                                f"<b>Model Family:</b> {row['model_family']}<br>" +
                                f"<b>Hardware:</b> {row['hardware_type']}<br>" +
                                f"<b>Batch Size:</b> {row['batch_size']}<br>" +
                                f"<b>Precision:</b> {row['precision']}<br>" +
                                f"<b>Throughput:</b> {row['throughput_items_per_second']:.2f} items/s<br>" +
                                f"<b>Energy:</b> {row['energy_consumption_joules']:.2f} joules<br>" +
                                f"<b>Latency:</b> {row['average_latency_ms']:.2f} ms<br>" +
                                f"<b>Efficiency:</b> {row['efficiency']:.2f} items/joule"
                                for _, row in hw_df.iterrows()
                            ],
                            hoverinfo="text",
                            name=f"{hw_type}",
                            visible=False  # Initially hidden
                        ))
                        
                        # Add a button to show only this hardware type
                        hardware_buttons.append(
                            dict(
                                args=[{"visible": [False] * (len(fig.data) - 1) + [True]}],
                                label=hw_type,
                                method="restyle"
                            )
                        )
                    
                    # Add dropdown menu
                    dropdowns.append(
                        dict(
                            buttons=hardware_buttons,
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.4,
                            xanchor="left",
                            y=1.1,
                            yanchor="top",
                            bgcolor=self.colors["bg_color"],
                            bordercolor=self.colors["accent_color"],
                            type="dropdown",
                            font=dict(color=self.colors["text_color"]),
                        )
                    )
                
                # Add all controls to the layout
                if dropdowns:
                    fig.update_layout(updatemenus=dropdowns)
                
                # Add annotations for the dropdowns
                if "model_family" in df.columns and len(df["model_family"].unique()) > 1:
                    fig.add_annotation(
                        x=0.05,
                        y=1.1,
                        xref="paper",
                        yref="paper",
                        text="Model Family:",
                        showarrow=False,
                        font=dict(size=12),
                        align="left",
                    )
                
                if "hardware_type" in df.columns and len(df["hardware_type"].unique()) > 1:
                    fig.add_annotation(
                        x=0.35,
                        y=1.1,
                        xref="paper",
                        yref="paper",
                        text="Hardware Type:",
                        showarrow=False,
                        font=dict(size=12),
                        align="left",
                    )
            
            # Save visualization to HTML file
            pio.write_html(
                fig,
                file=output_path,
                auto_open=self.config["auto_open"],
                include_plotlyjs='cdn',
            )
            
            # Optionally save data for later use
            if self.config["save_data"]:
                data_path = output_path.replace('.html', '_data.json')
                with open(data_path, 'w') as f:
                    json.dump({
                        "data_points": df.to_dict(orient="records"),
                        "metrics": ["throughput_items_per_second", "energy_consumption_joules", "average_latency_ms", "efficiency"],
                        "dimensions": ["hardware_type", "model_family", "batch_size", "precision"]
                    }, f, indent=2, default=str)
            
            logger.info(f"Power efficiency visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating power efficiency visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def create_animated_time_series_visualization(self,
                                             data: Optional[Dict[str, Any]] = None,
                                             metric: str = "throughput_items_per_second",
                                             dimensions: List[str] = ["model_family", "hardware_type"],
                                             time_range: Optional[int] = None,  # Days to look back
                                             time_interval: str = "day",  # 'hour', 'day', 'week', 'month'
                                             include_trend: bool = True,
                                             window_size: int = 5,  # Window size for moving average
                                             output_path: Optional[str] = None,
                                             title: str = "Performance Over Time") -> Optional[str]:
        """Create an animated time series visualization of performance metrics over time.
        
        This visualization shows how performance metrics change over time with animation controls,
        trend analysis, and anomaly detection.
        
        Args:
            data: Optional performance data dictionary (will use db_api if None)
            metric: The metric to visualize
            dimensions: Dimensions to group by (e.g., model_family, hardware_type)
            time_range: Number of days to look back (None for all time)
            time_interval: Time interval for aggregation ('hour', 'day', 'week', 'month')
            include_trend: Whether to include trend line
            window_size: Window size for moving average trend line
            output_path: Optional path for saving the visualization
            title: Title for the visualization
            
        Returns:
            Path to the saved visualization HTML file, or None if error
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.error("Plotly and Pandas are required for time series visualization.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"time_series_{timestamp}.html")
        
        try:
            # Get data if not provided
            if data is None and self.db_api:
                # Build query to get time series data
                query = """
                SELECT 
                    m.model_id,
                    m.model_name,
                    m.model_family,
                    hp.hardware_id,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.precision,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb,
                    pr.energy_consumption_joules,
                    pr.is_simulated,
                    pr.created_at
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                """
                
                params = {}
                where_clauses = []
                
                # Apply time range filter if specified
                if time_range is not None:
                    cutoff_date = datetime.now() - timedelta(days=time_range)
                    where_clauses.append("pr.created_at >= :cutoff_date")
                    params["cutoff_date"] = cutoff_date.isoformat()
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                
                query += """
                ORDER BY
                    pr.created_at
                """
                
                # Execute query
                results = self.db_api.query(query, params)
                
                if results.empty:
                    logger.error("No data found for the specified parameters.")
                    return None
                
                # Convert results to the required format
                data = {
                    "data_points": results.to_dict(orient="records")
                }
            
            if not data or "data_points" not in data:
                logger.error("No data available for visualization.")
                return None
            
            # Convert data to DataFrame
            df = pd.DataFrame(data["data_points"])
            
            if len(df) == 0:
                logger.error("No data points available for visualization.")
                return None
            
            # Ensure created_at is datetime
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
            else:
                logger.error("Time column 'created_at' not found in data.")
                return None
            
            # Identify unique timestamps for animation frames
            if time_interval == 'hour':
                df['time_bucket'] = df['created_at'].dt.floor('H')
            elif time_interval == 'day':
                df['time_bucket'] = df['created_at'].dt.floor('D')
            elif time_interval == 'week':
                df['time_bucket'] = df['created_at'].dt.floor('W')
            elif time_interval == 'month':
                df['time_bucket'] = df['created_at'].dt.floor('M')
            else:
                df['time_bucket'] = df['created_at'].dt.floor('D')  # Default to day
            
            time_buckets = sorted(df['time_bucket'].unique())
            
            # Determine metric to plot
            if metric == "throughput":
                metric_col = "throughput_items_per_second"
                metric_title = "Throughput (items/sec)"
            elif metric == "latency":
                metric_col = "average_latency_ms"
                metric_title = "Latency (ms)"
            elif metric == "memory":
                metric_col = "memory_peak_mb"
                metric_title = "Memory Usage (MB)"
            elif metric in df.columns:
                metric_col = metric
                metric_title = metric.replace('_', ' ').title()
            else:
                logger.error(f"Unknown metric: {metric}")
                return None
            
            # Create animation figure
            fig = go.Figure()
            
            # Define a colormap for different dimension combinations
            colors = px.colors.qualitative.Plotly
            
            # Create a unique color mapping for each dimension combination
            dimension_combinations = {}
            color_idx = 0
            
            # Calculate aggregated values for each dimension combination at each time bucket
            for time_bucket in time_buckets:
                time_df = df[df['time_bucket'] <= time_bucket]
                
                # Group by dimensions and get latest values for each group
                group_cols = dimensions + ['time_bucket']
                latest_df = time_df.sort_values('created_at').groupby(group_cols).last().reset_index()
                
                # For each unique dimension combination, add a trace
                for _, group in latest_df.groupby(dimensions):
                    # Create a unique key for this dimension combination
                    dim_key = '-'.join([str(group[dim].iloc[0]) for dim in dimensions])
                    
                    # Assign a consistent color to this dimension combination
                    if dim_key not in dimension_combinations:
                        dimension_combinations[dim_key] = {
                            'color': colors[color_idx % len(colors)],
                            'label': ' / '.join([f"{dim}={group[dim].iloc[0]}" for dim in dimensions])
                        }
                        color_idx += 1
                    
                    # Calculate trend line if requested
                    if include_trend and len(group) >= window_size:
                        group = group.sort_values('time_bucket')
                        group[f'{metric_col}_trend'] = group[metric_col].rolling(window=window_size, min_periods=1).mean()
                    
                    # Add a trace for this dimension combination
                    fig.add_trace(
                        go.Scatter(
                            x=group['time_bucket'],
                            y=group[metric_col],
                            mode='lines+markers',
                            name=dimension_combinations[dim_key]['label'],
                            line=dict(color=dimension_combinations[dim_key]['color']),
                            marker=dict(color=dimension_combinations[dim_key]['color']),
                            text=[
                                f"<b>Time:</b> {row['time_bucket']}<br>" +
                                ''.join([f"<b>{dim}:</b> {row[dim]}<br>" for dim in dimensions]) +
                                f"<b>{metric_title}:</b> {row[metric_col]:.2f}<br>" +
                                (f"<b>Trend:</b> {row[f'{metric_col}_trend']:.2f}" if include_trend and f'{metric_col}_trend' in group.columns else "")
                                for _, row in group.iterrows()
                            ],
                            hoverinfo="text",
                            visible=False  # Initially hidden, will be shown in animation
                        )
                    )
                    
                    # Add trend line if requested
                    if include_trend and f'{metric_col}_trend' in group.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=group['time_bucket'],
                                y=group[f'{metric_col}_trend'],
                                mode='lines',
                                line=dict(
                                    color=dimension_combinations[dim_key]['color'],
                                    width=1,
                                    dash='dash'
                                ),
                                name=f"{dimension_combinations[dim_key]['label']} (Trend)",
                                hoverinfo="skip",
                                visible=False  # Initially hidden, will be shown in animation
                            )
                        )
            
            # Make the first frame visible
            for i in range(len(fig.data) // len(time_buckets)):
                fig.data[i].visible = True
            
            # Create slider steps for animation
            steps = []
            for i, time_bucket in enumerate(time_buckets):
                step = dict(
                    method="update",
                    args=[
                        {"visible": [False] * len(fig.data)},  # Hide all traces
                        {"title": f"{title} - {time_bucket.strftime('%Y-%m-%d')}"} # Update title with timestamp
                    ],
                    label=time_bucket.strftime("%Y-%m-%d")
                )
                
                # Show traces for this time bucket and earlier
                for j in range(i + 1):
                    trace_idx_start = j * (len(fig.data) // len(time_buckets))
                    trace_idx_end = (j + 1) * (len(fig.data) // len(time_buckets))
                    for k in range(trace_idx_start, trace_idx_end):
                        step["args"][0]["visible"][k] = True
                
                steps.append(step)
            
            # Add slider for animation
            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Time: "},
                pad={"t": 50},
                steps=steps
            )]
            
            # Add play button for animation
            updatemenus = [dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[None, {"frame": {"duration": self.config["animation_duration"], "redraw": True}, "fromcurrent": True}],
                        label="Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        label="Pause",
                        method="animate"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )]
            
            # Update layout
            fig.update_layout(
                title=f"{title} - {time_buckets[0].strftime('%Y-%m-%d')}",
                xaxis=dict(
                    title="Time",
                    gridcolor=self.colors["grid_color"],
                    zeroline=False,
                ),
                yaxis=dict(
                    title=metric_title,
                    gridcolor=self.colors["grid_color"],
                    zeroline=False,
                ),
                width=self.config["default_width"],
                height=self.config["default_height"],
                template="plotly_white" if self.config["theme"] == "light" else "plotly_dark",
                sliders=sliders,
                updatemenus=updatemenus,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=30, t=50, b=60),
            )
            
            # Create animation frames
            frames = []
            for i, time_bucket in enumerate(time_buckets):
                frame_traces = []
                
                # Add all traces up to this time bucket
                for j in range(i + 1):
                    trace_idx_start = j * (len(fig.data) // len(time_buckets))
                    trace_idx_end = (j + 1) * (len(fig.data) // len(time_buckets))
                    for k in range(trace_idx_start, trace_idx_end):
                        frame_traces.append(fig.data[k])
                
                frame = go.Frame(
                    data=frame_traces,
                    name=time_bucket.strftime("%Y-%m-%d"),
                    traces=list(range(len(frame_traces)))
                )
                frames.append(frame)
            
            fig.frames = frames
            
            # Save visualization to HTML file
            pio.write_html(
                fig,
                file=output_path,
                auto_open=self.config["auto_open"],
                include_plotlyjs='cdn',
                animation_opts=dict(
                    frame=dict(duration=self.config["animation_duration"]),
                    transition=dict(duration=500),
                )
            )
            
            # Optionally save data for later use
            if self.config["save_data"]:
                data_path = output_path.replace('.html', '_data.json')
                with open(data_path, 'w') as f:
                    json.dump({
                        "data_points": df.to_dict(orient="records"),
                        "metric": metric,
                        "dimensions": dimensions,
                        "time_buckets": [tb.isoformat() for tb in time_buckets]
                    }, f, indent=2, default=str)
            
            logger.info(f"Animated time series visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating animated time series visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def create_dashboard(self,
                        dashboard_name: str,
                        title: Optional[str] = None,
                        description: Optional[str] = None,
                        template: Optional[str] = None,
                        components: Optional[List[Dict[str, Any]]] = None,
                        output_path: Optional[str] = None,
                        columns: Optional[int] = None,
                        row_height: Optional[int] = None) -> Optional[str]:
        """Create a customizable dashboard with multiple visualization components.
        
        Args:
            dashboard_name: Unique name for the dashboard (used for saving/loading)
            title: Dashboard title (defaults to dashboard_name if None)
            description: Optional description for the dashboard
            template: Optional template name to use as starting point ('overview', 'hardware_comparison', 'model_analysis', 'empty')
            components: List of component configurations to include in the dashboard
            output_path: Optional path for saving the dashboard HTML file
            columns: Number of columns in the dashboard grid (default from config)
            row_height: Height of each row in pixels (default from config)
            
        Returns:
            Path to the saved dashboard HTML file, or None if error
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly is required for dashboard creation.")
            return None
            
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.dashboards_dir, f"dashboard_{dashboard_name}_{timestamp}.html")
            
        # Use template if provided and available
        dashboard_config = {
            "name": dashboard_name,
            "title": title or dashboard_name.replace('_', ' ').title(),
            "description": description or "",
            "components": [],
            "columns": columns or self.config["dashboard_columns"],
            "row_height": row_height or self.config["dashboard_row_height"],
            "creation_date": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        if template and template in self.dashboard_templates:
            # Copy template configuration
            template_config = self.dashboard_templates[template]
            dashboard_config["title"] = title or template_config["title"]
            dashboard_config["description"] = description or template_config["description"]
            dashboard_config["components"] = template_config["components"].copy()
        
        # Add or override with provided components
        if components:
            dashboard_config["components"] = components
            
        try:
            # Create HTML file for the dashboard
            dashboard_html = self._generate_dashboard_html(dashboard_config)
            
            # Save dashboard HTML
            with open(output_path, 'w') as f:
                f.write(dashboard_html)
                
            # Save dashboard configuration
            self.saved_dashboards[dashboard_name] = dashboard_config
            self._save_dashboard_configs()
            
            # Open in browser if auto_open is enabled
            if self.config["auto_open"]:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
            logger.info(f"Dashboard '{dashboard_name}' created at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def list_dashboards(self) -> Dict[str, Dict[str, Any]]:
        """List all saved dashboard configurations.
        
        Returns:
            Dictionary of dashboard configurations indexed by name
        """
        return self.saved_dashboards
        
    def get_dashboard(self, dashboard_name: str) -> Optional[Dict[str, Any]]:
        """Get a saved dashboard configuration by name.
        
        Args:
            dashboard_name: Name of the dashboard to retrieve
            
        Returns:
            Dashboard configuration dictionary, or None if not found
        """
        return self.saved_dashboards.get(dashboard_name)
        
    def update_dashboard(self,
                       dashboard_name: str,
                       title: Optional[str] = None,
                       description: Optional[str] = None,
                       components: Optional[List[Dict[str, Any]]] = None,
                       columns: Optional[int] = None,
                       row_height: Optional[int] = None,
                       output_path: Optional[str] = None) -> Optional[str]:
        """Update an existing dashboard configuration.
        
        Args:
            dashboard_name: Name of the dashboard to update
            title: New title for the dashboard (if None, keep existing)
            description: New description for the dashboard (if None, keep existing)
            components: New list of component configurations (if None, keep existing)
            columns: New number of columns (if None, keep existing)
            row_height: New row height (if None, keep existing)
            output_path: Optional path for saving the updated dashboard HTML file
            
        Returns:
            Path to the saved dashboard HTML file, or None if error
        """
        if dashboard_name not in self.saved_dashboards:
            logger.error(f"Dashboard '{dashboard_name}' not found.")
            return None
            
        # Get existing configuration
        dashboard_config = self.saved_dashboards[dashboard_name].copy()
        
        # Update configuration with new values
        if title is not None:
            dashboard_config["title"] = title
            
        if description is not None:
            dashboard_config["description"] = description
            
        if components is not None:
            dashboard_config["components"] = components
            
        if columns is not None:
            dashboard_config["columns"] = columns
            
        if row_height is not None:
            dashboard_config["row_height"] = row_height
            
        dashboard_config["last_modified"] = datetime.now().isoformat()
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.dashboards_dir, f"dashboard_{dashboard_name}_{timestamp}.html")
            
        try:
            # Create HTML file for the dashboard
            dashboard_html = self._generate_dashboard_html(dashboard_config)
            
            # Save dashboard HTML
            with open(output_path, 'w') as f:
                f.write(dashboard_html)
                
            # Save updated dashboard configuration
            self.saved_dashboards[dashboard_name] = dashboard_config
            self._save_dashboard_configs()
            
            # Open in browser if auto_open is enabled
            if self.config["auto_open"]:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
            logger.info(f"Dashboard '{dashboard_name}' updated at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def delete_dashboard(self, dashboard_name: str) -> bool:
        """Delete a saved dashboard configuration.
        
        Args:
            dashboard_name: Name of the dashboard to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if dashboard_name not in self.saved_dashboards:
            logger.error(f"Dashboard '{dashboard_name}' not found.")
            return False
            
        try:
            # Remove dashboard from saved configurations
            del self.saved_dashboards[dashboard_name]
            self._save_dashboard_configs()
            
            logger.info(f"Dashboard '{dashboard_name}' deleted.")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting dashboard: {e}")
            return False
            
    def add_component_to_dashboard(self,
                                dashboard_name: str,
                                component_type: str,
                                component_config: Dict[str, Any],
                                width: int = 1,
                                height: int = 1,
                                position: Optional[Dict[str, int]] = None,
                                output_path: Optional[str] = None) -> Optional[str]:
        """Add a visualization component to a dashboard.
        
        Args:
            dashboard_name: Name of the dashboard to update
            component_type: Type of component to add ('3d', 'heatmap', 'power', 'time-series')
            component_config: Configuration for the component
            width: Width of the component in grid columns
            height: Height of the component in grid rows
            position: Optional position for the component ({"row": row, "col": col})
            output_path: Optional path for saving the updated dashboard HTML file
            
        Returns:
            Path to the saved dashboard HTML file, or None if error
        """
        if dashboard_name not in self.saved_dashboards:
            logger.error(f"Dashboard '{dashboard_name}' not found.")
            return None
            
        # Get existing configuration
        dashboard_config = self.saved_dashboards[dashboard_name].copy()
        
        # Prepare component configuration
        component = {
            "type": component_type,
            "width": width,
            "height": height,
            **component_config
        }
        
        # Add position if provided
        if position:
            component["position"] = position
            
        # Add component to dashboard
        dashboard_config["components"].append(component)
        dashboard_config["last_modified"] = datetime.now().isoformat()
        
        # Update dashboard
        return self.update_dashboard(
            dashboard_name=dashboard_name,
            components=dashboard_config["components"],
            output_path=output_path
        )
        
    def remove_component_from_dashboard(self,
                                       dashboard_name: str,
                                       component_index: int,
                                       output_path: Optional[str] = None) -> Optional[str]:
        """Remove a visualization component from a dashboard.
        
        Args:
            dashboard_name: Name of the dashboard to update
            component_index: Index of the component to remove
            output_path: Optional path for saving the updated dashboard HTML file
            
        Returns:
            Path to the saved dashboard HTML file, or None if error
        """
        if dashboard_name not in self.saved_dashboards:
            logger.error(f"Dashboard '{dashboard_name}' not found.")
            return None
            
        # Get existing configuration
        dashboard_config = self.saved_dashboards[dashboard_name].copy()
        
        # Check if component index is valid
        if component_index < 0 or component_index >= len(dashboard_config["components"]):
            logger.error(f"Invalid component index: {component_index}")
            return None
            
        # Remove component
        dashboard_config["components"].pop(component_index)
        dashboard_config["last_modified"] = datetime.now().isoformat()
        
        # Update dashboard
        return self.update_dashboard(
            dashboard_name=dashboard_name,
            components=dashboard_config["components"],
            output_path=output_path
        )
        
    def export_dashboard(self,
                       dashboard_name: str,
                       format: str = "html",
                       output_path: Optional[str] = None) -> Optional[str]:
        """Export a dashboard to different formats.
        
        Args:
            dashboard_name: Name of the dashboard to export
            format: Export format ('html', 'pdf', 'png')
            output_path: Optional path for saving the exported file
            
        Returns:
            Path to the exported file, or None if error
        """
        if dashboard_name not in self.saved_dashboards:
            logger.error(f"Dashboard '{dashboard_name}' not found.")
            return None
            
        # Get dashboard configuration
        dashboard_config = self.saved_dashboards[dashboard_name]
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.dashboards_dir, f"dashboard_{dashboard_name}_{timestamp}.{format}")
            
        try:
            if format == "html":
                # Create HTML file for the dashboard
                dashboard_html = self._generate_dashboard_html(dashboard_config)
                
                # Save dashboard HTML
                with open(output_path, 'w') as f:
                    f.write(dashboard_html)
                    
            elif format == "pdf":
                # For PDF export, first generate HTML
                html_path = self.export_dashboard(dashboard_name, format="html")
                
                # Then convert HTML to PDF
                try:
                    from weasyprint import HTML
                    HTML(html_path).write_pdf(output_path)
                except ImportError:
                    logger.error("WeasyPrint is required for PDF export. Install with: pip install weasyprint")
                    return None
                    
            elif format == "png":
                # For PNG export, first generate HTML
                html_path = self.export_dashboard(dashboard_name, format="html")
                
                # Then convert HTML to PNG
                try:
                    from selenium import webdriver
                    from selenium.webdriver.chrome.options import Options
                    
                    options = Options()
                    options.add_argument("--headless")
                    options.add_argument("--disable-gpu")
                    options.add_argument("--no-sandbox")
                    
                    driver = webdriver.Chrome(options=options)
                    driver.get(f"file://{os.path.abspath(html_path)}")
                    driver.implicitly_wait(10)  # Wait for dashboard to load
                    
                    # Set window size to match dashboard dimensions
                    width = dashboard_config.get("columns", 2) * 800
                    height = len(dashboard_config.get("components", [])) * 600
                    driver.set_window_size(width, height)
                    
                    # Take screenshot
                    driver.save_screenshot(output_path)
                    driver.quit()
                    
                except ImportError:
                    logger.error("Selenium is required for PNG export. Install with: pip install selenium")
                    return None
                    
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            logger.info(f"Dashboard '{dashboard_name}' exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def _save_dashboard_configs(self):
        """Save all dashboard configurations to disk."""
        try:
            with open(self.dashboard_config_path, 'w') as f:
                json.dump(self.saved_dashboards, f, indent=2, default=str)
                
            logger.info(f"Saved {len(self.saved_dashboards)} dashboard configurations")
            
        except Exception as e:
            logger.error(f"Error saving dashboard configurations: {e}")
            
    def _generate_dashboard_html(self, dashboard_config: Dict[str, Any]) -> str:
        """Generate HTML for a dashboard.
        
        Args:
            dashboard_config: Dashboard configuration dictionary
            
        Returns:
            HTML string for the dashboard
        """
        # Extract dashboard configuration
        title = dashboard_config.get("title", "Performance Dashboard")
        description = dashboard_config.get("description", "")
        components = dashboard_config.get("components", [])
        columns = dashboard_config.get("columns", self.config["dashboard_columns"])
        row_height = dashboard_config.get("row_height", self.config["dashboard_row_height"])
        
        # Generate visualization components
        viz_components = []
        
        for i, component in enumerate(components):
            # Extract component configuration
            component_type = component.get("type")
            component_width = component.get("width", 1)
            component_height = component.get("height", 1)
            
            # Generate visualization based on type
            try:
                viz_path = None
                if component_type == "3d":
                    metrics = component.get("metrics", ["throughput", "latency", "memory_usage"])
                    dimensions = component.get("dimensions", ["model_family", "hardware_type"])
                    viz_path = self.create_3d_performance_visualization(
                        metrics=metrics,
                        dimensions=dimensions,
                        output_path=os.path.join(self.output_dir, f"dashboard_component_{i}_3d.html"),
                        title=component.get("title", "3D Performance Visualization")
                    )
                elif component_type == "heatmap":
                    metric = component.get("metric", "throughput")
                    model_families = component.get("model_families")
                    hardware_types = component.get("hardware_types")
                    viz_path = self.create_hardware_comparison_heatmap(
                        metric=metric,
                        model_families=model_families,
                        hardware_types=hardware_types,
                        output_path=os.path.join(self.output_dir, f"dashboard_component_{i}_heatmap.html"),
                        title=component.get("title", f"{metric.title()} Heatmap")
                    )
                elif component_type == "power":
                    hardware_types = component.get("hardware_types")
                    model_families = component.get("model_families")
                    batch_sizes = component.get("batch_sizes")
                    viz_path = self.create_power_efficiency_visualization(
                        hardware_types=hardware_types,
                        model_families=model_families,
                        batch_sizes=batch_sizes,
                        output_path=os.path.join(self.output_dir, f"dashboard_component_{i}_power.html"),
                        title=component.get("title", "Power Efficiency Visualization")
                    )
                elif component_type == "time-series":
                    metric = component.get("metric", "throughput_items_per_second")
                    dimensions = component.get("dimensions", ["model_family", "hardware_type"])
                    time_range = component.get("time_range")
                    include_trend = component.get("include_trend", True)
                    window_size = component.get("window_size", 3)
                    viz_path = self.create_animated_time_series_visualization(
                        metric=metric,
                        dimensions=dimensions,
                        time_range=time_range,
                        include_trend=include_trend,
                        window_size=window_size,
                        output_path=os.path.join(self.output_dir, f"dashboard_component_{i}_time_series.html"),
                        title=component.get("title", "Performance Over Time")
                    )
                
                if viz_path:
                    # Extract the visualization content to embed in dashboard
                    with open(viz_path, 'r') as f:
                        viz_content = f.read()
                    
                    # Extract the plotly div from the visualization file
                    import re
                    plotly_div_match = re.search(r'<div id=".*?">.*?</div>', viz_content, re.DOTALL)
                    plotly_script_match = re.search(r'<script type="text/javascript">.*?</script>', viz_content, re.DOTALL)
                    
                    if plotly_div_match and plotly_script_match:
                        plotly_div = plotly_div_match.group(0)
                        plotly_script = plotly_script_match.group(0)
                        
                        # Add to visualization components
                        viz_components.append({
                            "width": component_width,
                            "height": component_height,
                            "title": component.get("title", f"{component_type.title()} Visualization"),
                            "div": plotly_div,
                            "script": plotly_script
                        })
            except Exception as e:
                logger.error(f"Error generating visualization for component {i}: {e}")
        
        # Generate grid layout for the dashboard
        grid_layout = self._generate_grid_layout(viz_components, columns, row_height)
        
        # Generate dashboard HTML
        theme = self.config["theme"]
        colors = self.colors
        
        # Base HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: {colors["dashboard_bg"]};
                    color: {colors["text_color"]};
                }}
                .dashboard-header {{
                    background-color: {colors["header_bg"]};
                    padding: 20px;
                    margin-bottom: 20px;
                    border-bottom: 1px solid {colors["card_border"]};
                }}
                .dashboard-title {{
                    margin: 0;
                    font-size: 24px;
                }}
                .dashboard-description {{
                    margin-top: 10px;
                    font-size: 14px;
                    opacity: 0.8;
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat({columns}, 1fr);
                    grid-gap: {self.config["dashboard_padding"]}px;
                    padding: 20px;
                }}
                .dashboard-card {{
                    background-color: {colors["card_bg"]};
                    border: 1px solid {colors["card_border"]};
                    border-radius: 5px;
                    overflow: hidden;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .dashboard-card-header {{
                    background-color: {colors["header_bg"]};
                    padding: 10px 15px;
                    border-bottom: 1px solid {colors["card_border"]};
                }}
                .dashboard-card-title {{
                    margin: 0;
                    font-size: 16px;
                }}
                .dashboard-card-content {{
                    padding: 15px;
                    overflow: hidden;
                }}
                .dashboard-footer {{
                    text-align: center;
                    padding: 20px;
                    font-size: 12px;
                    opacity: 0.6;
                }}
                {grid_layout["css"]}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1 class="dashboard-title">{title}</h1>
                <div class="dashboard-description">{description}</div>
            </div>
            
            <div class="dashboard-grid">
                {grid_layout["html"]}
            </div>
            
            <div class="dashboard-footer">
                Generated by IPFS Accelerate Advanced Visualization System - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            
            <script>
                // Script for dashboard functionality
                document.addEventListener('DOMContentLoaded', function() {{
                    // Any dashboard-specific JavaScript goes here
                }});
                
                // Scripts for individual visualizations
                {grid_layout["scripts"]}
            </script>
        </body>
        </html>
        """
        
        return html_template
        
    def _generate_grid_layout(self, components: List[Dict[str, Any]], columns: int, row_height: int) -> Dict[str, str]:
        """Generate grid layout HTML, CSS, and scripts for dashboard components.
        
        Args:
            components: List of component configurations with width, height, title, div, and script
            columns: Number of columns in the grid
            row_height: Height of each row in pixels
            
        Returns:
            Dictionary with HTML, CSS, and scripts for the grid layout
        """
        # Simplified grid layout algorithm
        # This is a simple algorithm that places components in a grid
        # A more advanced algorithm could be implemented for better layout
        
        layout = {
            "html": "",
            "css": "",
            "scripts": ""
        }
        
        # Create CSS classes for component grid positions
        css_rules = []
        html_components = []
        scripts = []
        
        for i, component in enumerate(components):
            width = component.get("width", 1)
            height = component.get("height", 1)
            
            # Calculate grid span
            grid_span = min(width, columns)  # Limit width to number of columns
            
            # Create CSS class for this component
            css_class = f"component-{i}"
            css_rules.append(f"""
                .{css_class} {{
                    grid-column: span {grid_span};
                    height: {height * row_height}px;
                }}
            """)
            
            # Create HTML for this component
            html_components.append(f"""
                <div class="dashboard-card {css_class}">
                    <div class="dashboard-card-header">
                        <h3 class="dashboard-card-title">{component.get("title", "Visualization")}</h3>
                    </div>
                    <div class="dashboard-card-content">
                        {component.get("div", "")}
                    </div>
                </div>
            """)
            
            # Add script for this component
            scripts.append(component.get("script", ""))
        
        # Combine all CSS rules
        layout["css"] = "\n".join(css_rules)
        
        # Combine all HTML components
        layout["html"] = "\n".join(html_components)
        
        # Combine all scripts
        layout["scripts"] = "\n".join(scripts)
        
        return layout