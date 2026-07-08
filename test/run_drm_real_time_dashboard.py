#!/usr/bin/env python3
"""
DRM Real-Time Performance Metrics Dashboard Runner

This script launches the real-time performance metrics dashboard for the
Dynamic Resource Management (DRM) component of the Distributed Testing Framework.

Features:
- Real-time resource utilization visualization
- Live worker metrics
- Performance trend analysis with statistical regression detection
- Scaling decision history and visualization
- Automatic alerts for performance issues

Dependencies:
- dash and dash-bootstrap-components for the web interface
- plotly for interactive visualizations
- scipy and numpy for statistical analysis
- pandas for data manipulation
"""

import os
import sys
import argparse
import logging
import webbrowser
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("drm_dashboard_runner")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import dash
    except ImportError:
        missing_deps.append("dash")
    
    try:
        import dash_bootstrap_components
    except ImportError:
        missing_deps.append("dash-bootstrap-components")
    
    try:
        import plotly
    except ImportError:
        missing_deps.append("plotly")
    
    # Optional dependencies
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy (optional)")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas (optional)")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    return missing_deps

def get_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the DRM Real-Time Performance Dashboard")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8085, help="Port to bind to")
    parser.add_argument("--db-path", default="benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--theme", default="dark", choices=["light", "dark"], help="Dashboard theme")
    parser.add_argument("--update-interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--retention", type=int, default=60, help="Data retention window in minutes")
    parser.add_argument("--browser", action="store_true", help="Open dashboard in browser")
    parser.add_argument("--drm-url", help="URL of DRM coordinator (optional)")
    parser.add_argument("--api-key", help="API key for DRM coordinator (optional)")
    
    return parser.parse_args()

def main():
    """Run the dashboard with the specified arguments."""
    args = get_arguments()
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        logger.error("Missing dependencies:")
        for dep in missing_deps:
            if "optional" in dep:
                logger.warning(f"  - {dep}")
            else:
                logger.error(f"  - {dep}")
        
        if any(dep for dep in missing_deps if "optional" not in dep):
            logger.error("Please install required dependencies:")
            logger.error("pip install dash dash-bootstrap-components plotly numpy")
            sys.exit(1)
    
    try:
        # Import dashboard class
        from data.duckdb.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
        
        # Initialize DRM instance if URL and API key provided
        drm_instance = None
        if args.drm_url and args.api_key:
            try:
                from data.duckdb.distributed_testing.drm_api_client import DRMAPIClient
                logger.info(f"Connecting to DRM coordinator at {args.drm_url}")
                drm_instance = DRMAPIClient(args.drm_url, args.api_key)
            except ImportError:
                logger.warning("Could not import DRMAPIClient. Running in standalone mode.")
        
        if not drm_instance:
            logger.info("Running in standalone mode with simulated data")
            try:
                # Try to use mock DRM for testing
                from data.duckdb.distributed_testing.testing.mock_drm import MockDynamicResourceManager
                drm_instance = MockDynamicResourceManager()
                logger.info("Using mock DRM instance for testing")
            except ImportError:
                logger.warning("No DRM instance available. Some features will be limited.")
        
        # Create dashboard
        dashboard = DRMRealTimeDashboard(
            dynamic_resource_manager=drm_instance,
            db_path=args.db_path,
            port=args.port,
            update_interval=args.update_interval,
            retention_window=args.retention,
            debug=args.debug,
            theme=args.theme
        )
        
        logger.info(f"Starting dashboard at http://{args.host}:{args.port}")
        
        # Open browser if requested
        if args.browser:
            webbrowser.open(f"http://{args.host}:{args.port}")
        
        # Start dashboard
        dashboard.start()
        
    except ImportError as e:
        logger.error(f"Error importing dashboard: {e}")
        logger.error("Make sure the drm_real_time_dashboard.py file is in the correct path.")
        logger.error("Required dependencies: dash dash-bootstrap-components plotly numpy")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")