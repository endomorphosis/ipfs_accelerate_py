#!/usr/bin/env python3
"""
Test script for the Customizable Dashboard System.

This script demonstrates the functionality of the CustomizableDashboard class,
including creating dashboards from templates, adding components, and managing dashboards.
"""

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_test")

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)

# Try to import the CustomizableDashboard
try:
    from duckdb_api.visualization.advanced_visualization import CustomizableDashboard
    from duckdb_api.visualization.advanced_visualization.viz_animated_time_series import AnimatedTimeSeriesVisualization
    from duckdb_api.visualization.advanced_visualization.viz_3d import Visualization3D
    from duckdb_api.visualization.advanced_visualization.viz_heatmap import HardwareHeatmapVisualization
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing CustomizableDashboard: {e}")
    DASHBOARD_AVAILABLE = False

# Try to import plotly for interactive features
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False


def generate_sample_data(days=90, model_families=None, hardware_types=None):
    """Generate sample performance data for testing."""
    if model_families is None:
        model_families = ["transformers", "vision", "audio"]
    
    if hardware_types is None:
        hardware_types = ["nvidia_a100", "amd_mi250", "intel_arc", "apple_m2"]
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create empty list to store data
    data = []
    
    # Generate data for each model family and hardware type
    for model_family in model_families:
        for hardware_type in hardware_types:
            # Base performance characteristics by hardware
            base_throughput = {
                "nvidia_a100": 100,
                "amd_mi250": 85,
                "intel_arc": 70,
                "apple_m2": 50
            }.get(hardware_type, 50)
            
            base_latency = {
                "nvidia_a100": 10,
                "amd_mi250": 12,
                "intel_arc": 15,
                "apple_m2": 20
            }.get(hardware_type, 15)
            
            base_memory = {
                "nvidia_a100": 40,
                "amd_mi250": 50,
                "intel_arc": 35,
                "apple_m2": 30
            }.get(hardware_type, 40)
            
            # Adjustment by model family
            if model_family == "transformers":
                throughput_factor = 0.8
                latency_factor = 1.2
                memory_factor = 1.5
            elif model_family == "vision":
                throughput_factor = 1.0
                latency_factor = 1.0
                memory_factor = 1.0
            elif model_family == "audio":
                throughput_factor = 1.2
                latency_factor = 0.8
                memory_factor = 0.7
            else:
                throughput_factor = 1.0
                latency_factor = 1.0
                memory_factor = 1.0
            
            # Generate a trend over time with some randomness
            for i, date in enumerate(dates):
                # Trend factor (gradual improvement over time)
                trend_factor = 1.0 + (i / len(dates)) * 0.2
                
                # Add an anomaly for testing anomaly detection
                if i % 30 == 0:
                    anomaly_factor = 0.5  # 50% drop
                elif i % 15 == 0:
                    anomaly_factor = 1.5  # 50% spike
                else:
                    anomaly_factor = 1.0
                
                # Calculate metrics with randomness
                throughput = base_throughput * throughput_factor * trend_factor * anomaly_factor * (1 + random.uniform(-0.1, 0.1))
                latency = base_latency * latency_factor / trend_factor * (1 / anomaly_factor) * (1 + random.uniform(-0.1, 0.1))
                memory = base_memory * memory_factor * (1 + random.uniform(-0.05, 0.05))
                
                # Add to data list
                data.append({
                    "date": date,
                    "model_family": model_family,
                    "hardware_type": hardware_type,
                    "throughput_items_per_second": throughput,
                    "average_latency_ms": latency,
                    "memory_peak_mb": memory
                })
    
    # Convert to DataFrame
    return pd.DataFrame(data)


def test_dashboard_creation(output_dir="./dashboards", template=None, open_browser=True):
    """Test the creation of a dashboard with the given template."""
    if not DASHBOARD_AVAILABLE:
        logger.error("CustomizableDashboard is not available. Cannot run test.")
        return None
    
    logger.info(f"Testing dashboard creation with template: {template}")
    
    # Generate sample data
    df = generate_sample_data()
    
    # Create dashboard
    dashboard = CustomizableDashboard(theme="light", output_dir=output_dir)
    
    # Set dashboard parameters
    dashboard_name = f"test_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if template:
        # Create dashboard from template
        dashboard_path = dashboard.create_dashboard(
            dashboard_name=dashboard_name,
            template=template
        )
    else:
        # Create custom dashboard with all visualization types
        dashboard_path = dashboard.create_dashboard(
            dashboard_name=dashboard_name,
            title="Custom Test Dashboard",
            description="This is a custom dashboard with all visualization types",
            components=[
                {
                    "type": "3d",
                    "config": {
                        "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                        "dimensions": ["model_family", "hardware_type"],
                        "title": "3D Performance Visualization"
                    },
                    "width": 2,
                    "height": 1
                },
                {
                    "type": "heatmap",
                    "config": {
                        "metric": "throughput_items_per_second",
                        "title": "Hardware Throughput Comparison"
                    },
                    "width": 1,
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
                        "dimensions": ["model_family", "hardware_type"],
                        "time_range": 90,
                        "title": "Performance Trends Over Time"
                    },
                    "width": 2,
                    "height": 1
                }
            ],
            columns=2,
            row_height=500
        )
    
    if dashboard_path:
        logger.info(f"Dashboard created successfully: {dashboard_path}")
        
        # Open the dashboard in a browser if requested
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
            except Exception as e:
                logger.error(f"Error opening dashboard in browser: {e}")
    else:
        logger.error("Failed to create dashboard")
    
    return dashboard_path


def test_dashboard_management(output_dir="./dashboards", open_browser=True):
    """Test dashboard management features."""
    if not DASHBOARD_AVAILABLE:
        logger.error("CustomizableDashboard is not available. Cannot run test.")
        return None
    
    logger.info("Testing dashboard management features")
    
    # Create dashboard
    dashboard = CustomizableDashboard(theme="light", output_dir=output_dir)
    
    # List available templates
    templates = dashboard.list_available_templates()
    logger.info(f"Available templates: {list(templates.keys())}")
    
    # List available components
    components = dashboard.list_available_components()
    logger.info(f"Available components: {list(components.keys())}")
    
    # Create a dashboard from template
    dashboard_name = f"managed_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dashboard_path = dashboard.create_dashboard(
        dashboard_name=dashboard_name,
        template="overview",
        title="Managed Dashboard Test",
        description="This dashboard demonstrates management features"
    )
    
    if not dashboard_path:
        logger.error("Failed to create initial dashboard")
        return None
    
    # Add a component
    logger.info("Adding a component to the dashboard")
    dashboard_path = dashboard.add_component_to_dashboard(
        dashboard_name=dashboard_name,
        component_type="heatmap",
        component_config={
            "metric": "memory_peak_mb",
            "title": "Memory Usage Comparison"
        },
        width=2,
        height=1
    )
    
    # Update the dashboard
    logger.info("Updating dashboard properties")
    dashboard_path = dashboard.update_dashboard(
        dashboard_name=dashboard_name,
        title="Updated Managed Dashboard",
        columns=3
    )
    
    # List all dashboards
    dashboards = dashboard.list_dashboards()
    logger.info(f"Available dashboards: {list(dashboards.keys())}")
    
    # Get dashboard configuration
    config = dashboard.get_dashboard(dashboard_name)
    logger.info(f"Dashboard has {len(config.get('components', []))} components")
    
    # Export dashboard
    export_path = dashboard.export_dashboard(
        dashboard_name=dashboard_name,
        format="html",
        output_path=os.path.join(output_dir, f"{dashboard_name}_exported.html")
    )
    
    if export_path:
        logger.info(f"Dashboard exported successfully: {export_path}")
        
        # Open the exported dashboard in a browser if requested
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(export_path)}")
            except Exception as e:
                logger.error(f"Error opening exported dashboard in browser: {e}")
    else:
        logger.error("Failed to export dashboard")
    
    # Don't delete the dashboard as part of the test to allow inspection
    # logger.info("Deleting dashboard")
    # dashboard.delete_dashboard(dashboard_name)
    
    return dashboard_path


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test the Customizable Dashboard System")
    parser.add_argument("--template", type=str, choices=["overview", "hardware_comparison", "model_analysis", "empty", "all"], 
                        help="Dashboard template to use")
    parser.add_argument("--output-dir", type=str, default="./dashboards", 
                        help="Output directory for dashboard files")
    parser.add_argument("--management-test", action="store_true", 
                        help="Run dashboard management tests")
    parser.add_argument("--no-open", action="store_true", 
                        help="Don't open dashboards in browser")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.template == "all":
        # Test all templates
        for template in ["overview", "hardware_comparison", "model_analysis", "empty"]:
            test_dashboard_creation(args.output_dir, template, not args.no_open)
    elif args.template:
        # Test specific template
        test_dashboard_creation(args.output_dir, args.template, not args.no_open)
    else:
        # Test custom dashboard
        test_dashboard_creation(args.output_dir, None, not args.no_open)
    
    if args.management_test:
        test_dashboard_management(args.output_dir, not args.no_open)
    
    logger.info("All tests completed")


if __name__ == "__main__":
    main()