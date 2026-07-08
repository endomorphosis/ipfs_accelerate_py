#!/usr/bin/env python3
"""
Enhanced Visualization Dashboard Runner Script

This script runs the enhanced visualization dashboard with advanced regression detection
and visualization capabilities for the Distributed Testing Framework.

Features:
- Interactive performance metrics visualization
- Statistical regression detection
- Advanced regression visualization with confidence intervals and trend analysis
- Comparative regression analysis across metrics
- Export capabilities for reports and visualizations
"""

import os
import sys
import argparse
import logging
import anyio
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
    parser.add_argument("--no-regression", action="store_true", help="Disable regression detection")
    parser.add_argument("--export-format", choices=["html", "png", "svg", "pdf", "json"], 
                        help="Export format for visualizations")
    parser.add_argument("--regression-report", action="store_true", 
                        help="Generate comprehensive regression report after analysis")
    
    return parser.parse_args()

async def main():
    """Run the dashboard with the specified arguments."""
    args = get_arguments()
    
    try:
        # Import dashboard class
        from data.duckdb.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard
        
        # Check if the regression visualization module is available
        try:
            from data.duckdb.distributed_testing.dashboard.regression_visualization import RegressionVisualization
            has_regression_visualization = True
            logger.info("Enhanced regression visualization module is available")
        except ImportError:
            has_regression_visualization = False
            if not args.no_regression:
                logger.warning("Enhanced regression visualization module not found. Using basic visualization.")
        
        # Check if the DuckDB database exists
        if not os.path.exists(args.db_path):
            logger.warning(f"DuckDB database file not found: {args.db_path}")
            
            # Create parent directory if not exists
            db_dir = os.path.dirname(args.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created directory for database: {db_dir}")
                
            # Try to connect anyway (it might create a new database)
            logger.info(f"Attempting to create a new DuckDB database: {args.db_path}")
        
        # Create output directory if not exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create and start dashboard
        dashboard = EnhancedVisualizationDashboard(
            db_path=args.db_path,
            output_dir=args.output_dir,
            host=args.host,
            port=args.port,
            debug=args.debug,
            theme=args.theme,
            enable_regression_detection=not args.no_regression,
            enhanced_visualization=has_regression_visualization
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
        logger.error("Required dependencies: dash dash-bootstrap-components plotly pandas scipy duckdb aiohttp")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        anyio.run(main())
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")