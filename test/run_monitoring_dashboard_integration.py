#!/usr/bin/env python3
"""
Run script for integrating Advanced Visualization System with Monitoring Dashboard.

This script synchronizes visualizations with the monitoring dashboard and can set up
automatic synchronization for continuous updates.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_monitoring_dashboard_integration")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Check for required dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    logger.warning("Requests library not found. Install with: pip install requests")
    HAS_REQUESTS = False

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    logger.warning("WebSocket client not found. Install with: pip install websocket-client")
    HAS_WEBSOCKET = False

# Import dashboard integration
try:
    from data.duckdb.visualization.advanced_visualization.monitor_dashboard_integration import (
        MonitorDashboardIntegration
    )
    HAS_DASHBOARD_INTEGRATION = True
except ImportError as e:
    logger.error(f"Error importing dashboard integration: {e}")
    HAS_DASHBOARD_INTEGRATION = False

# Import database API and visualization system
try:
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
    from data.duckdb.visualization.advanced_visualization import AdvancedVisualizationSystem
    HAS_VISUALIZATION_SYSTEM = True
except ImportError as e:
    logger.error(f"Error importing visualization system: {e}")
    HAS_VISUALIZATION_SYSTEM = False


def synchronize_visualizations(
    visualization_dir, 
    dashboard_url=None, 
    api_key=None, 
    file_pattern="*.html",
    recursive=True
):
    """
    Synchronize visualizations with the monitoring dashboard.
    
    Args:
        visualization_dir: Directory containing visualization files
        dashboard_url: URL of the monitoring dashboard
        api_key: API key for authentication
        file_pattern: Pattern for visualization files
        recursive: Whether to search directories recursively
        
    Returns:
        Number of synchronized visualizations
    """
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return 0
    
    # Check if visualization directory exists
    if not os.path.exists(visualization_dir):
        logger.error(f"Visualization directory not found: {visualization_dir}")
        return 0
    
    # Initialize dashboard integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Test connection
    if not integration.connect():
        logger.error(f"Failed to connect to dashboard at {dashboard_url}")
        return 0
    
    # Synchronize visualizations
    num_synced = integration.synchronize_visualizations(
        visualization_dir=visualization_dir,
        file_pattern=file_pattern,
        recursive=recursive
    )
    
    logger.info(f"Synchronized {num_synced} visualizations with the dashboard")
    return num_synced


def setup_auto_sync(
    visualization_dir, 
    dashboard_url=None, 
    api_key=None, 
    interval_seconds=60,
    max_runtime_minutes=None
):
    """
    Set up automatic synchronization with the monitoring dashboard.
    
    Args:
        visualization_dir: Directory containing visualization files
        dashboard_url: URL of the monitoring dashboard
        api_key: API key for authentication
        interval_seconds: Interval between synchronization runs
        max_runtime_minutes: Maximum runtime in minutes (None for indefinite)
    """
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return
    
    # Check if visualization directory exists
    if not os.path.exists(visualization_dir):
        logger.error(f"Visualization directory not found: {visualization_dir}")
        return
    
    # Initialize dashboard integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Test connection
    if not integration.connect():
        logger.error(f"Failed to connect to dashboard at {dashboard_url}")
        return
    
    # Set up auto-sync
    integration.setup_auto_sync(
        visualization_dir=visualization_dir,
        interval_seconds=interval_seconds,
        file_pattern="*.html",
        recursive=True,
        max_runtime_minutes=max_runtime_minutes
    )
    
    # Keep the script running
    if max_runtime_minutes is None:
        logger.info("Auto-sync is running. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Auto-sync stopped by user.")
    else:
        import time
        runtime_seconds = max_runtime_minutes * 60
        logger.info(f"Auto-sync will run for {max_runtime_minutes} minutes.")
        time.sleep(runtime_seconds)
        logger.info(f"Auto-sync completed after {max_runtime_minutes} minutes.")


def create_dashboard_panel(
    visualization_dir, 
    dashboard_url=None, 
    api_key=None,
    panel_title="Visualization Panel",
    dashboard_id="main"
):
    """
    Create a dashboard panel with visualizations from a directory.
    
    Args:
        visualization_dir: Directory containing visualization files
        dashboard_url: URL of the monitoring dashboard
        api_key: API key for authentication
        panel_title: Title for the panel
        dashboard_id: ID of the dashboard to add the panel to
        
    Returns:
        Panel ID if successful, None otherwise
    """
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return None
    
    # Check if visualization directory exists
    if not os.path.exists(visualization_dir):
        logger.error(f"Visualization directory not found: {visualization_dir}")
        return None
    
    # Initialize dashboard integration
    integration = MonitorDashboardIntegration(dashboard_url=dashboard_url, api_key=api_key)
    
    # Test connection
    if not integration.connect():
        logger.error(f"Failed to connect to dashboard at {dashboard_url}")
        return None
    
    # Find visualization files
    import glob
    viz_files = glob.glob(os.path.join(visualization_dir, "**", "*.html"), recursive=True)
    
    if not viz_files:
        logger.error(f"No visualization files found in {visualization_dir}")
        return None
    
    # Synchronize visualizations
    num_synced = integration.synchronize_visualizations(
        visualization_dir=visualization_dir,
        file_pattern="*.html",
        recursive=True
    )
    
    if num_synced == 0:
        logger.error("Failed to synchronize any visualizations")
        return None
    
    # Extract visualization IDs
    viz_ids = [os.path.splitext(os.path.basename(f))[0] for f in viz_files]
    
    # Create layout (grid)
    columns = min(2, len(viz_ids))  # Max 2 columns
    rows = (len(viz_ids) + columns - 1) // columns  # Ceiling division
    
    layout = {
        "type": "grid",
        "columns": columns,
        "rows": rows,
        "items": []
    }
    
    # Add items to layout
    for i, viz_id in enumerate(viz_ids):
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
        panel_title=panel_title,
        visualization_ids=viz_ids,
        layout=layout,
        dashboard_id=dashboard_id
    )
    
    if panel_id:
        logger.info(f"Created dashboard panel '{panel_title}' with {len(viz_ids)} visualizations")
    else:
        logger.error("Failed to create dashboard panel")
    
    return panel_id


def run_visualization_and_sync(
    db_path,
    output_dir,
    dashboard_url=None,
    api_key=None,
    num_visualizations=5,
    create_panel=True,
    auto_sync=False,
    interval_seconds=60,
    max_runtime_minutes=None
):
    """
    Run visualizations and synchronize with the dashboard.
    
    Args:
        db_path: Path to DuckDB database
        output_dir: Directory to save visualizations
        dashboard_url: URL of the monitoring dashboard
        api_key: API key for authentication
        num_visualizations: Number of visualizations to create
        create_panel: Whether to create a dashboard panel
        auto_sync: Whether to set up automatic synchronization
        interval_seconds: Interval between synchronization runs
        max_runtime_minutes: Maximum runtime in minutes (None for indefinite)
    """
    if not HAS_VISUALIZATION_SYSTEM:
        logger.error("Visualization system not available.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system
    vis = AdvancedVisualizationSystem(db_api=db_api, output_dir=output_dir)
    vis.configure({"auto_open": False})
    
    # Create visualizations
    logger.info(f"Creating {num_visualizations} visualizations...")
    viz_paths = []
    
    # Create 3D visualization
    for i in range(min(2, num_visualizations)):
        viz_path = vis.create_3d_performance_visualization(
            metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
            dimensions=["model_family", "hardware_type"],
            title=f"3D Performance Visualization {i+1}"
        )
        if viz_path:
            viz_paths.append(viz_path)
    
    # Create heatmap visualization
    for i in range(min(2, num_visualizations)):
        viz_path = vis.create_hardware_comparison_heatmap(
            metric="throughput",
            batch_size=1,
            title=f"Hardware Comparison Heatmap {i+1}"
        )
        if viz_path:
            viz_paths.append(viz_path)
    
    # Create time-series visualization
    for i in range(min(1, num_visualizations)):
        viz_path = vis.create_animated_time_series_visualization(
            metric="throughput_items_per_second",
            dimensions=["model_family", "hardware_type"],
            include_trend=True,
            window_size=3,
            title=f"Time Series Visualization {i+1}"
        )
        if viz_path:
            viz_paths.append(viz_path)
    
    if not viz_paths:
        logger.error("Failed to create any visualizations")
        return
    
    logger.info(f"Created {len(viz_paths)} visualizations")
    
    # Synchronize with dashboard
    num_synced = synchronize_visualizations(
        visualization_dir=output_dir,
        dashboard_url=dashboard_url,
        api_key=api_key
    )
    
    if num_synced == 0:
        logger.error("Failed to synchronize any visualizations with dashboard")
        return
    
    # Create dashboard panel
    if create_panel:
        panel_id = create_dashboard_panel(
            visualization_dir=output_dir,
            dashboard_url=dashboard_url,
            api_key=api_key,
            panel_title="Performance Visualization Panel"
        )
        
        if panel_id:
            logger.info(f"Created dashboard panel with ID: {panel_id}")
    
    # Set up automatic synchronization
    if auto_sync:
        setup_auto_sync(
            visualization_dir=output_dir,
            dashboard_url=dashboard_url,
            api_key=api_key,
            interval_seconds=interval_seconds,
            max_runtime_minutes=max_runtime_minutes
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitoring Dashboard Integration")
    parser.add_argument("--dashboard-url", default="http://localhost:8082",
                      help="URL of the monitoring dashboard")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                      help="Path to the DuckDB database")
    parser.add_argument("--output-dir", default="./advanced_visualizations",
                      help="Directory containing visualizations")
    parser.add_argument("--mode", choices=["sync", "auto-sync", "panel", "run-and-sync"],
                      default="sync", help="Operation mode")
    parser.add_argument("--interval", type=int, default=60,
                      help="Interval between synchronization runs (seconds)")
    parser.add_argument("--runtime", type=int, help="Maximum runtime in minutes (auto-sync only)")
    parser.add_argument("--num-viz", type=int, default=5,
                      help="Number of visualizations to create (run-and-sync only)")
    
    args = parser.parse_args()
    
    # Check required dependencies
    missing_deps = []
    if not HAS_REQUESTS:
        missing_deps.append("requests")
    if not HAS_WEBSOCKET:
        missing_deps.append("websocket-client")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error(f"Please install with: pip install {' '.join(missing_deps)}")
        return 1
    
    if not HAS_DASHBOARD_INTEGRATION:
        logger.error("Dashboard integration not available.")
        return 1
    
    # Perform requested operation
    if args.mode == "sync":
        # Synchronize visualizations with dashboard
        num_synced = synchronize_visualizations(
            visualization_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key
        )
        
        if num_synced == 0:
            logger.error("Failed to synchronize any visualizations")
            return 1
        
    elif args.mode == "auto-sync":
        # Set up automatic synchronization
        setup_auto_sync(
            visualization_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key,
            interval_seconds=args.interval,
            max_runtime_minutes=args.runtime
        )
        
    elif args.mode == "panel":
        # Create dashboard panel
        panel_id = create_dashboard_panel(
            visualization_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key
        )
        
        if not panel_id:
            logger.error("Failed to create dashboard panel")
            return 1
        
    elif args.mode == "run-and-sync":
        # Run visualizations and synchronize with dashboard
        run_visualization_and_sync(
            db_path=args.db_path,
            output_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key,
            num_visualizations=args.num_viz,
            create_panel=True,
            auto_sync=(args.interval > 0),
            interval_seconds=args.interval,
            max_runtime_minutes=args.runtime
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())