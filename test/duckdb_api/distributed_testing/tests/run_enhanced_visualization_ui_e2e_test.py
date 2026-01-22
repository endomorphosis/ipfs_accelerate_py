#!/usr/bin/env python3
"""
End-to-End Test Runner for the Enhanced Visualization UI Features.

This script provides a comprehensive end-to-end test of the enhanced UI functionality:
1. Sets up a temporary test database with performance data containing known regressions
2. Launches the dashboard with enhanced visualization enabled
3. Tests the visualization options panel and export functionality
4. Verifies the integration between visualization options and rendering
5. Tests theme integration between dashboard and visualizations

Usage:
    python run_enhanced_visualization_ui_e2e_test.py [--port PORT] [--no-browser] [--debug]
"""

import os
import sys
import argparse
import logging
import tempfile
import time
import webbrowser
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_enhanced_visualization_ui_e2e_test")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import necessary components
try:
    from duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard
    from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector
    from duckdb_api.distributed_testing.dashboard.regression_visualization import RegressionVisualization
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    from duckdb_api.distributed_testing.tests.test_dashboard_regression_integration import generate_performance_data_with_regressions
    HAS_REQUIRED_COMPONENTS = True
except ImportError as e:
    logger.error(f"Error importing required components: {e}")
    HAS_REQUIRED_COMPONENTS = False


def run_dashboard(db_path, output_dir, port, debug=False, open_browser=True):
    """Run the Enhanced Visualization Dashboard with enhanced UI features enabled."""
    logger.info(f"Starting Enhanced Visualization Dashboard with Enhanced UI on port {port}...")
    
    # Create a database connection
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Create dashboard with regression detection and enhanced visualization
    dashboard = EnhancedVisualizationDashboard(
        db_conn=db_api,
        output_dir=output_dir,
        enable_regression_detection=True,
        enhanced_visualization=True,
        debug=debug
    )
    
    # Configure host and port
    host = '0.0.0.0' if debug else 'localhost'
    
    # Display information
    logger.info(f"Dashboard URL: http://{host}:{port}")
    logger.info("Enhanced UI Features Test Instructions:")
    logger.info("\n1. Navigate to the 'Regression Detection' tab")
    logger.info("2. Select 'latency_ms' from the dropdown and filter for 'bert-base' model and 'webgpu' hardware")
    logger.info("3. Run regression analysis to detect the regression around Jan 31, 2025")
    logger.info("4. In the visualization options panel, try toggling the following options on/off:")
    logger.info("   - Show Confidence Intervals")
    logger.info("   - Show Trend Lines")
    logger.info("   - Show Annotations")
    logger.info("   Notice how the visualization updates in real-time")
    logger.info("5. Select different export formats (HTML, PNG, SVG, JSON, PDF) and test the export buttons")
    logger.info("6. Generate a regression report and verify it contains the visualization options")
    logger.info("7. Try changing the theme (light/dark) in the settings panel and verify the visualization updates")
    logger.info("\nPress Ctrl+C to stop the dashboard when testing is complete")
    
    # Open browser if requested
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    
    # Run the dashboard
    dashboard.run_server(
        host=host,
        port=port,
        debug=debug
    )


def main():
    """Main function to run the end-to-end test."""
    if not HAS_REQUIRED_COMPONENTS:
        logger.error("Required components not available. Please check your installation.")
        return 1
    
    parser = argparse.ArgumentParser(description="Run end-to-end test for Enhanced Visualization UI features")
    parser.add_argument("--port", type=int, default=8083, help="Port to run the dashboard on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output-dir", help="Output directory for visualizations (default: temporary directory)")
    parser.add_argument("--db-path", help="Path to DuckDB database (default: temporary file)")
    
    args = parser.parse_args()
    
    # Create a temporary directory if not specified
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        cleanup_temp_dir = False
    else:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
        cleanup_temp_dir = True
    
    # Create a temporary database if not specified
    if args.db_path:
        db_path = args.db_path
        cleanup_temp_db = False
    else:
        db_path = os.path.join(output_dir, "test_benchmark.duckdb")
        cleanup_temp_db = True
    
    try:
        # Generate test data if database doesn't exist or is empty
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            logger.info(f"Generating test data in {db_path}...")
            generate_performance_data_with_regressions(db_path)
            logger.info("Test data generation complete.")
        
        # Run the dashboard with enhanced UI features
        run_dashboard(
            db_path=db_path,
            output_dir=output_dir,
            port=args.port,
            debug=args.debug,
            open_browser=not args.no_browser
        )
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
        return 0
    
    except Exception as e:
        logger.error(f"Error in end-to-end test: {e}", exc_info=True)
        return 1
    
    finally:
        # Clean up temporary files if we created them
        if cleanup_temp_dir and 'temp_dir' in locals():
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())