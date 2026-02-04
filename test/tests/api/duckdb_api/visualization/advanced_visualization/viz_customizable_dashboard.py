"""
Customizable Dashboard System for the Advanced Visualization System.

This module provides a customizable dashboard system that allows combining multiple
visualization components into interactive dashboards with flexible layouts.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import shutil

from test.tests.api.duckdb_api.visualization.advanced_visualization.base import BaseVisualization, PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_visualization")

# Import optional dependencies
if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

# Try to import visualization components
try:
    from test.tests.api.duckdb_api.visualization.advanced_visualization.viz_3d import Visualization3D
    from test.tests.api.duckdb_api.visualization.advanced_visualization.viz_heatmap import HardwareHeatmapVisualization
    from test.tests.api.duckdb_api.visualization.advanced_visualization.viz_time_series import TimeSeriesVisualization
    from test.tests.api.duckdb_api.visualization.advanced_visualization.viz_animated_time_series import AnimatedTimeSeriesVisualization
    COMPONENTS_AVAILABLE = True
except ImportError:
    logger.warning("One or more visualization components not available.")
    COMPONENTS_AVAILABLE = False


class CustomizableDashboard(BaseVisualization):
    """
    Customizable Dashboard System that combines multiple visualization components.
    
    This class allows creating interactive dashboards with flexible layouts,
    combining various visualization components like 3D visualizations, heatmaps,
    time-series plots, and more.
    """
    
    def __init__(self, db_connection=None, theme="light", debug=False, output_dir="./dashboards"):
        """Initialize the dashboard with database connection, theme, and output directory."""
        super().__init__(db_connection, theme, debug)
        self.output_dir = output_dir
        self.components = []
        self.layout = {"columns": 2, "row_height": 500}
        self.dashboard_config = {}
        self.dashboard_name = None
        self.title = "Performance Dashboard"
        self.description = "Customizable dashboard for performance visualization"
        self.dashboard_dir = None
        self.component_registry = self._initialize_component_registry()
        self.dashboard_templates = self._initialize_dashboard_templates()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to store dashboard configurations
        self.config_dir = os.path.join(output_dir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
    
    def _initialize_component_registry(self):
        """Initialize the registry of available dashboard components."""
        registry = {}
        
        if COMPONENTS_AVAILABLE:
            registry.update({
                "3d": {
                    "class": Visualization3D,
                    "description": "3D visualization for exploring multi-dimensional data",
                    "parameters": ["metrics", "dimensions", "filters", "title"]
                },
                "heatmap": {
                    "class": HardwareHeatmapVisualization,
                    "description": "Heatmap visualization for comparing hardware performance",
                    "parameters": ["metric", "model_families", "hardware_types", "title"]
                },
                "time-series": {
                    "class": TimeSeriesVisualization,
                    "description": "Time-series visualization for tracking metrics over time",
                    "parameters": ["metric", "dimensions", "time_range", "title"]
                },
                "animated-time-series": {
                    "class": AnimatedTimeSeriesVisualization,
                    "description": "Animated time-series visualization with interactive controls",
                    "parameters": ["metric", "dimensions", "time_range", "title", "events"]
                }
            })
        
        return registry
    
    def _initialize_dashboard_templates(self):
        """Initialize predefined dashboard templates."""
        templates = {
            "overview": {
                "title": "Performance Overview Dashboard",
                "description": "General overview of performance metrics across models and hardware",
                "columns": 2,
                "row_height": 500,
                "components": [
                    {
                        "type": "3d",
                        "config": {
                            "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                            "dimensions": ["model_family", "hardware_type"],
                            "title": "3D Performance Visualization"
                        },
                        "width": 1,
                        "height": 1
                    },
                    {
                        "type": "heatmap",
                        "config": {
                            "metric": "throughput_items_per_second",
                            "title": "Hardware Comparison Heatmap"
                        },
                        "width": 1,
                        "height": 1
                    },
                    {
                        "type": "animated-time-series",
                        "config": {
                            "metric": "throughput_items_per_second",
                            "dimensions": ["model_family", "hardware_type"],
                            "time_range": 90,
                            "title": "Performance Trends Over Time"
                        },
                        "width": 2,
                        "height": 1
                    }
                ]
            },
            "hardware_comparison": {
                "title": "Hardware Comparison Dashboard",
                "description": "Detailed comparison of hardware platforms",
                "columns": 2,
                "row_height": 500,
                "components": [
                    {
                        "type": "heatmap",
                        "config": {
                            "metric": "throughput_items_per_second",
                            "title": "Hardware Throughput Comparison"
                        },
                        "width": 2,
                        "height": 1
                    },
                    {
                        "type": "heatmap",
                        "config": {
                            "metric": "average_latency_ms",
                            "title": "Hardware Latency Comparison"
                        },
                        "width": 1,
                        "height": 1
                    },
                    {
                        "type": "animated-time-series",
                        "config": {
                            "metric": "throughput_items_per_second",
                            "dimensions": ["hardware_type"],
                            "time_range": 90,
                            "title": "Hardware Performance Trends"
                        },
                        "width": 1,
                        "height": 1
                    }
                ]
            },
            "model_analysis": {
                "title": "Model Analysis Dashboard",
                "description": "Detailed analysis of model performance",
                "columns": 2,
                "row_height": 500,
                "components": [
                    {
                        "type": "3d",
                        "config": {
                            "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                            "dimensions": ["model_family"],
                            "title": "Model Performance in 3D"
                        },
                        "width": 1,
                        "height": 1
                    },
                    {
                        "type": "animated-time-series",
                        "config": {
                            "metric": "throughput_items_per_second",
                            "dimensions": ["model_family"],
                            "time_range": 90,
                            "title": "Model Performance Trends"
                        },
                        "width": 1,
                        "height": 1
                    },
                    {
                        "type": "heatmap",
                        "config": {
                            "metric": "average_latency_ms",
                            "title": "Model Latency Comparison"
                        },
                        "width": 2,
                        "height": 1
                    }
                ]
            },
            "empty": {
                "title": "Empty Dashboard",
                "description": "A blank dashboard template",
                "columns": 2,
                "row_height": 500,
                "components": []
            }
        }
        
        return templates
    
    def list_available_components(self):
        """List all available component types for dashboards."""
        return {comp_type: info["description"] for comp_type, info in self.component_registry.items()}
    
    def list_available_templates(self):
        """List all available dashboard templates."""
        return {name: {
            "title": template["title"],
            "description": template["description"],
            "components": len(template["components"])
        } for name, template in self.dashboard_templates.items()}
    
    def create_dashboard(self, dashboard_name=None, template=None, title=None, description=None, 
                        components=None, columns=None, row_height=None):
        """
        Create a new dashboard based on a template or custom configuration.
        
        Args:
            dashboard_name (str): Unique name for the dashboard
            template (str): Optional template name to use as starting point
            title (str): Dashboard title
            description (str): Dashboard description
            components (list): List of component configurations
            columns (int): Number of columns in the grid layout
            row_height (int): Height of each row in pixels
            
        Returns:
            str: Path to the created dashboard HTML
        """
        # Generate a dashboard name if not provided
        if dashboard_name is None:
            dashboard_name = f"dashboard_{uuid.uuid4().hex[:8]}"
        
        self.dashboard_name = dashboard_name
        
        # Create directory for dashboard assets
        self.dashboard_dir = os.path.join(self.output_dir, dashboard_name)
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Initialize dashboard configuration
        self.dashboard_config = {
            "name": dashboard_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "components": []
        }
        
        # Use template if specified
        if template is not None:
            if template not in self.dashboard_templates:
                raise ValueError(f"Template '{template}' not found. Available templates: {list(self.dashboard_templates.keys())}")
            
            template_config = self.dashboard_templates[template]
            self.title = template_config["title"] if title is None else title
            self.description = template_config["description"] if description is None else description
            self.layout["columns"] = template_config["columns"] if columns is None else columns
            self.layout["row_height"] = template_config["row_height"] if row_height is None else row_height
            self.components = template_config["components"].copy() if components is None else components
        else:
            # Use custom configuration
            self.title = title if title is not None else "Custom Dashboard"
            self.description = description if description is not None else "Custom dashboard configuration"
            self.layout["columns"] = columns if columns is not None else 2
            self.layout["row_height"] = row_height if row_height is not None else 500
            self.components = components if components is not None else []
        
        # Update dashboard configuration
        self.dashboard_config.update({
            "title": self.title,
            "description": self.description,
            "layout": self.layout,
            "components": self.components
        })
        
        # Create dashboard HTML
        dashboard_path = self._generate_dashboard_html()
        
        # Save dashboard configuration
        self._save_dashboard_config()
        
        return dashboard_path
    
    def _generate_dashboard_html(self):
        """Generate the dashboard HTML file combining all components."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly is required for dashboard generation")
            return None
        
        # Create sub-folders for component outputs
        components_dir = os.path.join(self.dashboard_dir, "components")
        os.makedirs(components_dir, exist_ok=True)
        
        # Generate each component and collect their HTML
        component_html = []
        component_paths = []
        
        for idx, component_config in enumerate(self.components):
            component_type = component_config["type"]
            config = component_config.get("config", {})
            width = component_config.get("width", 1)
            height = component_config.get("height", 1)
            
            # Skip if component type not available
            if component_type not in self.component_registry:
                logger.warning(f"Component type '{component_type}' not available. Skipping.")
                continue
            
            # Create component instance
            component_class = self.component_registry[component_type]["class"]
            component = component_class(self.db_connection, self.theme, self.debug)
            
            # Determine appropriate creation method based on component type
            if component_type == "3d":
                creation_method = getattr(component, "create_3d_visualization", None)
                if creation_method:
                    result = creation_method(**config)
                else:
                    logger.warning(f"Component {component_type} does not have create_3d_visualization method")
                    continue
            elif component_type == "heatmap":
                creation_method = getattr(component, "create_hardware_heatmap", None)
                if creation_method:
                    result = creation_method(**config)
                else:
                    logger.warning(f"Component {component_type} does not have create_hardware_heatmap method")
                    continue
            elif component_type == "time-series":
                creation_method = getattr(component, "create_time_series_visualization", None)
                if creation_method:
                    result = creation_method(**config)
                else:
                    logger.warning(f"Component {component_type} does not have create_time_series_visualization method")
                    continue
            elif component_type == "animated-time-series":
                creation_method = getattr(component, "create_animated_time_series", None)
                if creation_method:
                    result = creation_method(**config)
                else:
                    logger.warning(f"Component {component_type} does not have create_animated_time_series method")
                    continue
            else:
                logger.warning(f"Unknown component type: {component_type}")
                continue
            
            # Save component to file
            component_filename = f"component_{idx}.html"
            component_path = os.path.join(components_dir, component_filename)
            
            try:
                # Get the figure and save it
                fig = component.figure
                fig.write_html(component_path, include_plotlyjs="cdn", full_html=False)
                
                # Read the saved component HTML
                with open(component_path, 'r') as f:
                    component_content = f.read()
                
                # Add to component list with size information
                component_html.append({
                    "content": component_content,
                    "width": width,
                    "height": height,
                    "type": component_type,
                    "title": config.get("title", f"Component {idx}")
                })
                
                component_paths.append(component_path)
            except Exception as e:
                logger.error(f"Error generating component {idx} of type {component_type}: {e}")
                continue
        
        # Generate the main dashboard HTML
        dashboard_html = self._generate_dashboard_layout(component_html)
        
        # Write dashboard HTML to file
        dashboard_path = os.path.join(self.dashboard_dir, "dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        return dashboard_path
    
    def _generate_dashboard_layout(self, components):
        """Generate the HTML layout for the dashboard with components."""
        columns = self.layout["columns"]
        row_height = self.layout["row_height"]
        
        # CSS for dashboard layout
        dashboard_css = f"""
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: {self.theme_colors["background"]};
                color: {self.theme_colors["text"]};
            }}
            .dashboard-header {{
                padding: 20px;
                text-align: center;
                background-color: {self.theme_colors["accent1"]};
                color: white;
            }}
            .dashboard-description {{
                padding: 10px 20px;
                margin-bottom: 20px;
                border-bottom: 1px solid {self.theme_colors["grid"]};
            }}
            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat({columns}, 1fr);
                gap: 20px;
                padding: 20px;
            }}
            .dashboard-component {{
                background-color: {self.theme_colors["background"]};
                border: 1px solid {self.theme_colors["grid"]};
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .dashboard-component-header {{
                padding: 10px;
                font-weight: bold;
                background-color: {self.theme_colors["accent1"] + "20"};
                border-bottom: 1px solid {self.theme_colors["grid"]};
            }}
            .dashboard-component-content {{
                padding: 10px;
                height: calc({row_height}px - 40px);
                overflow: auto;
            }}
            .dashboard-footer {{
                padding: 10px 20px;
                text-align: center;
                font-size: 0.8em;
                border-top: 1px solid {self.theme_colors["grid"]};
                margin-top: 20px;
            }}
            
            /* Style for Plotly interactive components */
            .js-plotly-plot {{
                width: 100%;
                height: 100%;
            }}
        </style>
        """
        
        # Header HTML
        header_html = f"""
        <div class="dashboard-header">
            <h1>{self.title}</h1>
        </div>
        <div class="dashboard-description">
            <p>{self.description}</p>
        </div>
        """
        
        # Grid HTML for components
        grid_html = '<div class="dashboard-grid">'
        
        for idx, component in enumerate(components):
            width = component["width"]
            height = component["height"]
            title = component["title"]
            content = component["content"]
            
            # CSS grid span
            grid_style = f'style="grid-column: span {width}; grid-row: span {height};"'
            
            # Component HTML
            grid_html += f"""
            <div class="dashboard-component" {grid_style}>
                <div class="dashboard-component-header">{title}</div>
                <div class="dashboard-component-content">
                    {content}
                </div>
            </div>
            """
        
        grid_html += '</div>'
        
        # Footer HTML
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_html = f"""
        <div class="dashboard-footer">
            <p>Generated on {current_date} | IPFS Accelerate Advanced Visualization System</p>
        </div>
        """
        
        # Complete HTML document
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            {dashboard_css}
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            {header_html}
            {grid_html}
            {footer_html}
        </body>
        </html>
        """
        
        return dashboard_html
    
    def _save_dashboard_config(self):
        """Save the dashboard configuration to a JSON file."""
        config_path = os.path.join(self.config_dir, f"{self.dashboard_name}.json")
        
        with open(config_path, 'w') as f:
            json.dump(self.dashboard_config, f, indent=4)
        
        return config_path
    
    def list_dashboards(self):
        """List all saved dashboards with their metadata."""
        dashboards = {}
        
        for config_file in os.listdir(self.config_dir):
            if config_file.endswith('.json'):
                try:
                    with open(os.path.join(self.config_dir, config_file), 'r') as f:
                        config = json.load(f)
                    
                    dashboard_name = config.get("name", config_file.replace('.json', ''))
                    dashboards[dashboard_name] = {
                        "title": config.get("title", "Untitled Dashboard"),
                        "description": config.get("description", ""),
                        "components": len(config.get("components", [])),
                        "created_at": config.get("created_at", ""),
                        "updated_at": config.get("updated_at", "")
                    }
                except Exception as e:
                    logger.error(f"Error loading dashboard config {config_file}: {e}")
        
        return dashboards
    
    def get_dashboard(self, dashboard_name):
        """Get the configuration of a specific dashboard."""
        config_path = os.path.join(self.config_dir, f"{dashboard_name}.json")
        
        if not os.path.exists(config_path):
            raise ValueError(f"Dashboard '{dashboard_name}' not found")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading dashboard config {dashboard_name}: {e}")
            return None
    
    def update_dashboard(self, dashboard_name, title=None, description=None, columns=None, row_height=None):
        """Update an existing dashboard configuration."""
        # Load the current configuration
        current_config = self.get_dashboard(dashboard_name)
        if current_config is None:
            raise ValueError(f"Dashboard '{dashboard_name}' not found or could not be loaded")
        
        # Update the dashboard properties
        self.dashboard_name = dashboard_name
        self.title = title if title is not None else current_config.get("title", "Untitled Dashboard")
        self.description = description if description is not None else current_config.get("description", "")
        self.layout = current_config.get("layout", {"columns": 2, "row_height": 500})
        
        if columns is not None:
            self.layout["columns"] = columns
        if row_height is not None:
            self.layout["row_height"] = row_height
        
        self.components = current_config.get("components", [])
        
        # Update the dashboard configuration
        self.dashboard_config = current_config
        self.dashboard_config.update({
            "title": self.title,
            "description": self.description,
            "layout": self.layout,
            "updated_at": datetime.now().isoformat()
        })
        
        # Recreate the dashboard directory
        self.dashboard_dir = os.path.join(self.output_dir, dashboard_name)
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Generate updated dashboard HTML
        dashboard_path = self._generate_dashboard_html()
        
        # Save updated configuration
        self._save_dashboard_config()
        
        return dashboard_path
    
    def add_component_to_dashboard(self, dashboard_name, component_type, component_config, width=1, height=1):
        """Add a new component to an existing dashboard."""
        # Check if component type is valid
        if component_type not in self.component_registry:
            raise ValueError(f"Component type '{component_type}' not available. Available types: {list(self.component_registry.keys())}")
        
        # Load the current configuration
        current_config = self.get_dashboard(dashboard_name)
        if current_config is None:
            raise ValueError(f"Dashboard '{dashboard_name}' not found or could not be loaded")
        
        # Create new component configuration
        new_component = {
            "type": component_type,
            "config": component_config,
            "width": width,
            "height": height
        }
        
        # Add component to the configuration
        self.dashboard_name = dashboard_name
        self.title = current_config.get("title", "Untitled Dashboard")
        self.description = current_config.get("description", "")
        self.layout = current_config.get("layout", {"columns": 2, "row_height": 500})
        self.components = current_config.get("components", [])
        self.components.append(new_component)
        
        # Update the dashboard configuration
        self.dashboard_config = current_config
        self.dashboard_config.update({
            "components": self.components,
            "updated_at": datetime.now().isoformat()
        })
        
        # Recreate the dashboard directory
        self.dashboard_dir = os.path.join(self.output_dir, dashboard_name)
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Generate updated dashboard HTML
        dashboard_path = self._generate_dashboard_html()
        
        # Save updated configuration
        self._save_dashboard_config()
        
        return dashboard_path
    
    def remove_component_from_dashboard(self, dashboard_name, component_index):
        """Remove a component from an existing dashboard."""
        # Load the current configuration
        current_config = self.get_dashboard(dashboard_name)
        if current_config is None:
            raise ValueError(f"Dashboard '{dashboard_name}' not found or could not be loaded")
        
        # Check if component index is valid
        components = current_config.get("components", [])
        if component_index < 0 or component_index >= len(components):
            raise ValueError(f"Component index {component_index} is out of range (0-{len(components)-1})")
        
        # Remove the component
        self.dashboard_name = dashboard_name
        self.title = current_config.get("title", "Untitled Dashboard")
        self.description = current_config.get("description", "")
        self.layout = current_config.get("layout", {"columns": 2, "row_height": 500})
        self.components = components
        self.components.pop(component_index)
        
        # Update the dashboard configuration
        self.dashboard_config = current_config
        self.dashboard_config.update({
            "components": self.components,
            "updated_at": datetime.now().isoformat()
        })
        
        # Recreate the dashboard directory
        self.dashboard_dir = os.path.join(self.output_dir, dashboard_name)
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Generate updated dashboard HTML
        dashboard_path = self._generate_dashboard_html()
        
        # Save updated configuration
        self._save_dashboard_config()
        
        return dashboard_path
    
    def export_dashboard(self, dashboard_name, format="html", output_path=None):
        """Export a dashboard to different formats."""
        # Load the dashboard configuration
        current_config = self.get_dashboard(dashboard_name)
        if current_config is None:
            raise ValueError(f"Dashboard '{dashboard_name}' not found or could not be loaded")
        
        # Set up the dashboard properties
        self.dashboard_name = dashboard_name
        self.title = current_config.get("title", "Untitled Dashboard")
        self.description = current_config.get("description", "")
        self.layout = current_config.get("layout", {"columns": 2, "row_height": 500})
        self.components = current_config.get("components", [])
        self.dashboard_config = current_config
        
        # Set the dashboard directory
        self.dashboard_dir = os.path.join(self.output_dir, dashboard_name)
        
        # Determine output path
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{dashboard_name}.{format}")
        
        # Handle different export formats
        if format == "html":
            # The dashboard is already in HTML format, just copy it
            dashboard_html_path = os.path.join(self.dashboard_dir, "dashboard.html")
            if os.path.exists(dashboard_html_path):
                shutil.copy(dashboard_html_path, output_path)
                return output_path
            else:
                # Regenerate the dashboard HTML
                dashboard_path = self._generate_dashboard_html()
                if dashboard_path:
                    shutil.copy(dashboard_path, output_path)
                    return output_path
        
        elif format in ["png", "pdf"]:
            # For static formats, we need to use a tool like Playwright or wkhtmltopdf
            # This is a simplified implementation that relies on system tools
            
            # First, make sure we have an HTML version
            dashboard_html_path = os.path.join(self.dashboard_dir, "dashboard.html")
            if not os.path.exists(dashboard_html_path):
                dashboard_html_path = self._generate_dashboard_html()
            
            # Convert to desired format
            if format == "png":
                # Try using wkhtmltoimage if available
                try:
                    import subprocess
                    result = subprocess.run(
                        ["wkhtmltoimage", "--quality", "100", dashboard_html_path, output_path],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return output_path
                    else:
                        logger.error(f"Error exporting to PNG: {result.stderr}")
                        logger.error("Make sure wkhtmltoimage is installed")
                        return None
                except Exception as e:
                    logger.error(f"Error converting HTML to PNG: {e}")
                    logger.error("Alternative: Install wkhtmltoimage or take a screenshot manually")
                    return None
            
            elif format == "pdf":
                # Try using wkhtmltopdf if available
                try:
                    import subprocess
                    result = subprocess.run(
                        ["wkhtmltopdf", dashboard_html_path, output_path],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return output_path
                    else:
                        logger.error(f"Error exporting to PDF: {result.stderr}")
                        logger.error("Make sure wkhtmltopdf is installed")
                        return None
                except Exception as e:
                    logger.error(f"Error converting HTML to PDF: {e}")
                    logger.error("Alternative: Install wkhtmltopdf or print to PDF manually")
                    return None
        
        else:
            logger.error(f"Unsupported export format: {format}")
            logger.error("Supported formats: html, png, pdf")
            return None
    
    def delete_dashboard(self, dashboard_name):
        """Delete a dashboard and its configuration."""
        config_path = os.path.join(self.config_dir, f"{dashboard_name}.json")
        dashboard_dir = os.path.join(self.output_dir, dashboard_name)
        
        # Check if the dashboard exists
        if not os.path.exists(config_path):
            raise ValueError(f"Dashboard '{dashboard_name}' not found")
        
        # Delete the configuration file
        os.remove(config_path)
        
        # Delete the dashboard directory if it exists
        if os.path.exists(dashboard_dir) and os.path.isdir(dashboard_dir):
            shutil.rmtree(dashboard_dir)
        
        return True
    
    def create_visualization(self, **kwargs):
        """Create a dashboard with the given parameters."""
        return self.create_dashboard(**kwargs)