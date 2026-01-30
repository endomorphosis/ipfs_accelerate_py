#!/usr/bin/env python3
"""
Test script for the Monitoring Dashboard Integration module.

This script tests the integration between the Advanced Visualization System 
and the Monitoring Dashboard.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_dashboard_integration")

# Add parent directory to path for module imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import the dashboard integration module
try:
    from data.duckdb.visualization.advanced_visualization.monitor_dashboard_integration import (
        MonitorDashboardIntegration, 
        MonitorDashboardIntegrationMixin
    )
    HAS_DASHBOARD_INTEGRATION = True
except ImportError as e:
    logger.error(f"Error importing dashboard integration: {e}")
    logger.error("Make sure all required dependencies are installed.")
    HAS_DASHBOARD_INTEGRATION = False

# Import the visualization system
try:
    from data.duckdb.visualization.advanced_visualization import AdvancedVisualizationSystem
    HAS_ADVANCED_VISUALIZATION = True
except ImportError as e:
    logger.error(f"Error importing AdvancedVisualizationSystem: {e}")
    HAS_ADVANCED_VISUALIZATION = False

# Import database API
try:
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
    HAS_DB_API = True
except ImportError as e:
    logger.error(f"Error importing BenchmarkDBAPI: {e}")
    HAS_DB_API = False


def test_dashboard_connection(dashboard_url=None, api_key=None):
    """Test connection to the monitoring dashboard."""
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return False
    
    # Initialize integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Test connection
    return integration.connect()


def test_visualization_registration(dashboard_url=None, api_key=None, output_dir="./test_output"):
    """Test registering a visualization with the dashboard."""
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return False
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Create a simple HTML visualization
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Visualization</title>
    </head>
    <body>
        <h1>Test Visualization</h1>
        <div id="visualization">
            <p>This is a test visualization for the dashboard integration.</p>
            <svg width="200" height="100">
                <rect width="200" height="100" style="fill:rgb(0,0,255);stroke-width:3;stroke:rgb(0,0,0)" />
                <text x="50" y="50" fill="white">Test</text>
            </svg>
        </div>
    </body>
    </html>
    """
    
    # Save HTML to file
    html_path = os.path.join(output_dir, "test_visualization.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Register visualization
    success = integration.register_visualization(
        visualization_id="test_visualization",
        visualization_type="generic",
        metadata={
            "title": "Test Visualization",
            "description": "A test visualization for dashboard integration",
            "creation_time": datetime.now().isoformat()
        },
        html_content=html_content
    )
    
    return success


def create_test_visualizations(db_path, output_dir, num_visualizations=3):
    """Create test visualizations using the Advanced Visualization System."""
    if not HAS_ADVANCED_VISUALIZATION or not HAS_DB_API:
        logger.error("Advanced visualization or DB API not available.")
        return []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system
    # Note: Usually the MonitorDashboardIntegrationMixin would be mixed into 
    # AdvancedVisualizationSystem at class definition time, but for testing
    # we're just creating a simple object with the needed properties
    vis = AdvancedVisualizationSystem(db_api=db_api, output_dir=output_dir)
    vis.configure({"auto_open": False})
    
    # Create visualizations
    viz_paths = []
    for i in range(num_visualizations):
        viz_type = ["3d", "heatmap", "time-series"][i % 3]
        
        if viz_type == "3d":
            # Create 3D visualization
            viz_path = vis.create_3d_performance_visualization(
                metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                dimensions=["model_family", "hardware_type"],
                title=f"Test 3D Visualization {i+1}"
            )
        elif viz_type == "heatmap":
            # Create heatmap visualization
            viz_path = vis.create_hardware_comparison_heatmap(
                metric="throughput",
                batch_size=1,
                title=f"Test Heatmap Visualization {i+1}"
            )
        else:  # time-series
            # Create time-series visualization
            viz_path = vis.create_animated_time_series_visualization(
                metric="throughput_items_per_second",
                dimensions=["model_family", "hardware_type"],
                include_trend=True,
                window_size=3,
                title=f"Test Time Series Visualization {i+1}"
            )
        
        if viz_path:
            viz_paths.append(viz_path)
    
    return viz_paths


def test_visualization_sync(dashboard_url=None, api_key=None, db_path="./benchmark_db.duckdb", 
                           output_dir="./test_output"):
    """Test synchronizing visualizations with the dashboard."""
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return 0
    
    # Create test visualizations
    viz_paths = create_test_visualizations(db_path, output_dir)
    
    if not viz_paths:
        logger.error("Failed to create test visualizations.")
        return 0
    
    # Initialize integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Synchronize visualizations
    num_synced = integration.synchronize_visualizations(
        visualization_dir=output_dir,
        file_pattern="*.html",
        recursive=True
    )
    
    return num_synced


def test_dashboard_panels(dashboard_url=None, api_key=None, db_path="./benchmark_db.duckdb", 
                         output_dir="./test_output"):
    """Test creating dashboard panels with visualizations."""
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return False
    
    # Create test visualizations
    viz_paths = create_test_visualizations(db_path, output_dir)
    
    if not viz_paths:
        logger.error("Failed to create test visualizations.")
        return False
    
    # Initialize integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Synchronize visualizations
    viz_ids = integration.synchronize_visualizations(
        visualization_dir=output_dir,
        file_pattern="*.html",
        recursive=True
    )
    
    if not viz_ids:
        logger.error("Failed to synchronize visualizations.")
        return False
    
    # Extract visualization IDs (filenames without extension)
    visualization_ids = [os.path.splitext(os.path.basename(path))[0] for path in viz_paths]
    
    # Create panel layout (simple grid)
    num_items = len(visualization_ids)
    columns = min(2, num_items)  # 2 columns max
    rows = (num_items + columns - 1) // columns  # Ceiling division
    
    layout = {
        "type": "grid",
        "columns": columns,
        "rows": rows,
        "items": []
    }
    
    # Add items to layout
    for i, viz_id in enumerate(visualization_ids):
        row = i // columns
        col = i % columns
        
        layout["items"].append({
            "visualization_id": viz_id,
            "row": row,
            "col": col,
            "width": 1,
            "height": 1
        })
    
    # Create panel
    panel_id = integration.create_dashboard_panel(
        panel_title="Test Visualization Panel",
        visualization_ids=visualization_ids,
        layout=layout
    )
    
    return panel_id is not None


def test_dashboard_snapshot(dashboard_url=None, api_key=None, output_dir="./test_output"):
    """Test exporting and importing a dashboard snapshot."""
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return False
    
    # Initialize integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Export snapshot
    snapshot_path = integration.export_dashboard_snapshot(
        output_path=os.path.join(output_dir, "dashboard_snapshot.json")
    )
    
    if not snapshot_path:
        logger.error("Failed to export dashboard snapshot.")
        return False
    
    # Import snapshot (to a different dashboard ID)
    success = integration.import_dashboard_snapshot(
        snapshot_path=snapshot_path,
        target_dashboard_id="test_import"
    )
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Monitoring Dashboard Integration")
    parser.add_argument("--dashboard-url", default="http://localhost:8082",
                      help="URL of the monitoring dashboard")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                      help="Path to the DuckDB database")
    parser.add_argument("--output-dir", default="./test_dashboard_output",
                      help="Directory to save test outputs")
    parser.add_argument("--test", choices=["connection", "registration", "sync", "panels", "snapshot", "all"],
                      default="all", help="Test to run")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        logger.error("Please install the required dependencies:")
        logger.error("pip install requests websocket-client")
        return 1
    
    # Run the specified test
    if args.test == "connection" or args.test == "all":
        if test_dashboard_connection(args.dashboard_url, args.api_key):
            logger.info("✅ Connection test passed")
        else:
            logger.error("❌ Connection test failed")
    
    if args.test == "registration" or args.test == "all":
        if test_visualization_registration(args.dashboard_url, args.api_key, args.output_dir):
            logger.info("✅ Visualization registration test passed")
        else:
            logger.error("❌ Visualization registration test failed")
    
    if args.test == "sync" or args.test == "all":
        num_synced = test_visualization_sync(args.dashboard_url, args.api_key, args.db_path, args.output_dir)
        if num_synced > 0:
            logger.info(f"✅ Visualization sync test passed: {num_synced} visualizations synchronized")
        else:
            logger.error("❌ Visualization sync test failed")
    
    if args.test == "panels" or args.test == "all":
        if test_dashboard_panels(args.dashboard_url, args.api_key, args.db_path, args.output_dir):
            logger.info("✅ Dashboard panels test passed")
        else:
            logger.error("❌ Dashboard panels test failed")
    
    if args.test == "snapshot" or args.test == "all":
        if test_dashboard_snapshot(args.dashboard_url, args.api_key, args.output_dir):
            logger.info("✅ Dashboard snapshot test passed")
        else:
            logger.error("❌ Dashboard snapshot test failed")
    
    logger.info("Testing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())