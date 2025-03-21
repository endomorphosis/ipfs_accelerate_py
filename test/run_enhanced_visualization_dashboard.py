#!/usr/bin/env python3
"""
Enhanced Visualization Dashboard Runner Script

This script runs the enhanced visualization dashboard with regression detection capabilities.
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_runner")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def get_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the enhanced visualization dashboard with regression detection")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind to")
    parser.add_argument("--db-path", default="benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="./visualizations/dashboard", help="Output directory for visualizations")
    parser.add_argument("--theme", default="dark", choices=["light", "dark"], help="Dashboard theme")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--browser", action="store_true", help="Open dashboard in browser")
    
    return parser.parse_args()

async def main():
    """Run the dashboard with the specified arguments."""
    args = get_arguments()
    
    try:
        # Import dashboard class
        from duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard
        
        # Create and start dashboard
        dashboard = EnhancedVisualizationDashboard(
            db_path=args.db_path,
            output_dir=args.output_dir,
            host=args.host,
            port=args.port,
            debug=args.debug,
            theme=args.theme
        )
        
        logger.info(f"Starting dashboard at http://{args.host}:{args.port}")
        
        # Open browser if requested
        if args.browser:
            import webbrowser
            webbrowser.open(f"http://{args.host}:{args.port}")
        
        # Start dashboard
        await dashboard.start()
        
    except ImportError as e:
        logger.error(f"Error importing dashboard: {e}")
        logger.error("Make sure the enhanced_visualization_dashboard.py file is in the correct path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")