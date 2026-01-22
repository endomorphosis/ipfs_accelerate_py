#!/usr/bin/env python3
"""
Test script for the Advanced Visualization System.

This script demonstrates the capabilities of the Advanced Visualization System
by creating sample visualizations using test data.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_advanced_viz")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import database API
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI

# Import visualization system
try:
    from duckdb_api.visualization.advanced_visualization import AdvancedVisualizationSystem
    HAS_ADVANCED_VISUALIZATION = True
except ImportError as e:
    logger.error(f"Error importing AdvancedVisualizationSystem: {e}")
    logger.error("Make sure plotly, pandas, and scikit-learn are installed.")
    logger.error("Install with: pip install plotly pandas scikit-learn")
    HAS_ADVANCED_VISUALIZATION = False

def create_3d_visualization(db_path, output_dir, open_viz=True):
    """Create a 3D performance visualization."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization not available.")
        return
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=output_dir)
    viz.configure({"auto_open": open_viz})
    
    # Create 3D visualization
    metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
    dimensions = ["model_family", "hardware_type", "batch_size", "precision"] 
    
    viz_path = viz.create_3d_performance_visualization(
        metrics=metrics,
        dimensions=dimensions,
        title="3D Performance Visualization",
    )
    
    logger.info(f"3D visualization created: {viz_path}")
    return viz_path

def create_hardware_heatmap(db_path, output_dir, open_viz=True):
    """Create a hardware comparison heatmap by model family."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization not available.")
        return
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=output_dir)
    viz.configure({"auto_open": open_viz})
    
    # Create hardware comparison heatmap
    viz_path = viz.create_hardware_comparison_heatmap(
        metric="throughput",
        batch_size=1,
        title="Hardware Comparison by Model Family",
    )
    
    logger.info(f"Hardware comparison heatmap created: {viz_path}")
    return viz_path

def create_power_efficiency_viz(db_path, output_dir, open_viz=True):
    """Create a power efficiency visualization."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization not available.")
        return
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=output_dir)
    viz.configure({"auto_open": open_viz})
    
    # Create power efficiency visualization
    viz_path = viz.create_power_efficiency_visualization(
        title="Power Efficiency Visualization",
    )
    
    logger.info(f"Power efficiency visualization created: {viz_path}")
    return viz_path

def create_animated_time_series(db_path, output_dir, open_viz=True):
    """Create an animated time series visualization."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization not available.")
        return
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=output_dir)
    viz.configure({"auto_open": open_viz})
    
    # Create animated time series visualization
    viz_path = viz.create_animated_time_series_visualization(
        metric="throughput_items_per_second",
        dimensions=["model_family", "hardware_type"],
        include_trend=True,
        window_size=3,
        title="Performance Trends Over Time",
    )
    
    logger.info(f"Animated time series visualization created: {viz_path}")
    return viz_path

def create_dashboard(db_path, output_dir, template=None, open_viz=True):
    """Create a customizable dashboard with multiple visualization components."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization not available.")
        return
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=output_dir)
    viz.configure({"auto_open": open_viz})
    
    # Create dashboard
    dashboard_name = f"test_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if template:
        # Use predefined template
        dashboard_path = viz.create_dashboard(
            dashboard_name=dashboard_name,
            template=template,
            title=f"{template.replace('_', ' ').title()} Dashboard",
        )
    else:
        # Create custom dashboard with all visualization types
        dashboard_path = viz.create_dashboard(
            dashboard_name=dashboard_name,
            title="Comprehensive Performance Dashboard",
            description="This dashboard combines multiple visualization types to provide a comprehensive view of performance metrics.",
            components=[
                {
                    "type": "3d",
                    "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                    "dimensions": ["model_family", "hardware_type"],
                    "title": "3D Performance Visualization",
                    "width": 1,
                    "height": 1
                },
                {
                    "type": "heatmap",
                    "metric": "throughput",
                    "title": "Throughput Heatmap by Model Family",
                    "width": 1,
                    "height": 1
                },
                {
                    "type": "power",
                    "title": "Power Efficiency Analysis",
                    "width": 2,
                    "height": 1
                },
                {
                    "type": "time-series",
                    "metric": "throughput_items_per_second",
                    "dimensions": ["model_family", "hardware_type"],
                    "include_trend": True,
                    "window_size": 3,
                    "title": "Performance Trends Over Time",
                    "width": 2,
                    "height": 1
                }
            ],
            columns=2,
            row_height=500
        )
    
    logger.info(f"Dashboard created: {dashboard_path}")
    return dashboard_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Advanced Visualization System")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--output-dir", default="./advanced_visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--no-open", action="store_true",
                       help="Don't automatically open visualizations in browser")
    parser.add_argument("--viz-type", choices=["3d", "heatmap", "power", "time-series", "dashboard", "all"],
                       default="all", help="Type of visualization to create")
    parser.add_argument("--dashboard-template", choices=["overview", "hardware_comparison", "model_analysis", "empty"],
                       help="Template to use for dashboard creation")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization components not available.")
        logger.error("Please install the required dependencies:")
        logger.error("pip install plotly pandas scikit-learn")
        return 1
    
    # Check if the database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database file not found: {args.db_path}")
        logger.error("Please provide a valid database path.")
        return 1
    
    # Create visualizations based on the specified type
    if args.viz_type == "3d" or args.viz_type == "all":
        create_3d_visualization(args.db_path, args.output_dir, not args.no_open)
    
    if args.viz_type == "heatmap" or args.viz_type == "all":
        create_hardware_heatmap(args.db_path, args.output_dir, not args.no_open)
    
    if args.viz_type == "power" or args.viz_type == "all":
        create_power_efficiency_viz(args.db_path, args.output_dir, not args.no_open)
    
    if args.viz_type == "time-series" or args.viz_type == "all":
        create_animated_time_series(args.db_path, args.output_dir, not args.no_open)
    
    if args.viz_type == "dashboard" or args.viz_type == "all":
        create_dashboard(args.db_path, args.output_dir, args.dashboard_template, not args.no_open)
    
    logger.info("Visualization creation complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())