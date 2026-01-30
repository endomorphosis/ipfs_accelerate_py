#!/usr/bin/env python3
"""
Run Monitoring Dashboard with Error Visualization Integration.

This script runs the Distributed Testing Monitoring Dashboard with the Error Visualization
integration enabled, providing comprehensive error visualization, pattern detection, and analysis.

Usage:
    python -m duckdb_api.distributed_testing.run_monitoring_dashboard_with_error_visualization
"""

import argparse
import anyio
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_monitoring_dashboard_with_error_visualization")

# Add parent directory to path to import the dashboard
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.duckdb.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard

async def main():
    """Run the monitoring dashboard with error visualization integration."""
    parser = argparse.ArgumentParser(description="Run Monitoring Dashboard with Error Visualization")
    
    # General dashboard options
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--coordinator-url", help="URL of the coordinator service")
    parser.add_argument("--refresh-interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--theme", default="light", choices=["light", "dark"], help="Dashboard theme")
    
    # Database options
    parser.add_argument("--db-path", help="Path to DuckDB database file")
    
    # Directory options
    parser.add_argument("--static-dir", help="Directory for static files")
    parser.add_argument("--template-dir", help="Directory for HTML templates")
    parser.add_argument("--dashboard-dir", default="./dashboards", help="Directory to store dashboards")
    
    # Integration options
    parser.add_argument("--enable-result-aggregator", action="store_true", help="Enable Result Aggregator integration")
    parser.add_argument("--result-aggregator-url", help="URL of the result aggregator service")
    
    parser.add_argument("--enable-e2e-test", action="store_true", help="Enable E2E Test integration")
    
    parser.add_argument("--enable-visualization", action="store_true", help="Enable Visualization integration")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create dashboard with error visualization enabled
    dashboard = MonitoringDashboard(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator_url,
        result_aggregator_url=args.result_aggregator_url,
        refresh_interval=args.refresh_interval,
        static_dir=args.static_dir,
        template_dir=args.template_dir,
        theme=args.theme,
        enable_result_aggregator_integration=args.enable_result_aggregator,
        enable_e2e_test_integration=args.enable_e2e_test,
        enable_visualization_integration=args.enable_visualization,
        enable_error_visualization=True,  # Enable error visualization
        dashboard_dir=args.dashboard_dir,
        db_path=args.db_path
    )
    
    # Start the dashboard
    logger.info(f"Starting Monitoring Dashboard with Error Visualization on http://{args.host}:{args.port}")
    
    await dashboard.start()

if __name__ == "__main__":
    try:
        anyio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)