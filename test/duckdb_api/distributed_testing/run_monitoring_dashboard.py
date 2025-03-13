#!/usr/bin/env python3
"""
Run Monitoring Dashboard for Distributed Testing Framework

This script runs the comprehensive monitoring dashboard for the distributed 
testing framework, providing real-time monitoring of workers, tasks,
and system metrics.

Implementation Date: March 17, 2025 (Originally planned for June 19-26, 2025)

Usage:
    python run_monitoring_dashboard.py [options]

Options:
    --host HOST             Host to bind the server to (default: localhost)
    --port PORT             Port to bind the server to (default: 8082)
    --coordinator URL       URL of the coordinator server
    --theme THEME           Dashboard theme (light or dark, default: dark)
    --refresh SECONDS       Auto-refresh interval in seconds (default: 30, 0 to disable)
    --output-dir DIR        Output directory for dashboard files (default: ./monitoring_dashboard)
    --browser               Open dashboard in browser
    --no-alerts             Disable alert generation
    --real-time             Enable real-time updates via WebSockets
    --debug                 Enable debug logging
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import dashboard component
try:
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Import result aggregator if available
try:
    from duckdb_api.distributed_testing.result_aggregator.service import ResultAggregatorService
    RESULT_AGGREGATOR_AVAILABLE = True
except ImportError:
    RESULT_AGGREGATOR_AVAILABLE = False

# Import database manager if available
try:
    from duckdb_api.core.db_manager import BenchmarkDBManager
    DB_MANAGER_AVAILABLE = True
except ImportError:
    DB_MANAGER_AVAILABLE = False


def main():
    """Run the monitoring dashboard."""
    parser = argparse.ArgumentParser(description="Run Monitoring Dashboard for Distributed Testing")
    
    # General options
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind the server to")
    parser.add_argument("--coordinator", help="URL of the coordinator server")
    parser.add_argument("--output-dir", default="./monitoring_dashboard", 
                       help="Output directory for dashboard files")
    
    # Appearance options
    parser.add_argument("--theme", choices=["light", "dark"], default="dark", 
                       help="Dashboard theme")
    parser.add_argument("--refresh", type=int, default=30, 
                       help="Auto-refresh interval in seconds (0 to disable)")
    
    # Feature options
    parser.add_argument("--browser", action="store_true", 
                       help="Open dashboard in browser")
    parser.add_argument("--no-alerts", action="store_true", 
                       help="Disable alert generation")
    parser.add_argument("--real-time", action="store_true", 
                       help="Enable real-time updates via WebSockets")
    
    # Database options
    parser.add_argument("--db-path", 
                       help="Path to DuckDB database file (default: ./benchmark_db.duckdb)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    
    # Check dashboard availability
    if not DASHBOARD_AVAILABLE:
        print("Error: Monitoring dashboard is not available. Make sure all dependencies are installed.")
        sys.exit(1)
    
    # Create result aggregator if available
    result_aggregator = None
    if RESULT_AGGREGATOR_AVAILABLE:
        try:
            # First try to create database manager if needed
            db_manager = None
            if DB_MANAGER_AVAILABLE:
                db_path = args.db_path or "./benchmark_db.duckdb"
                db_manager = BenchmarkDBManager(db_path)
                print(f"Using database at: {db_path}")
            
            # Create result aggregator
            result_aggregator = ResultAggregatorService(db_manager=db_manager)
            print("Result aggregator initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize result aggregator: {e}")
            print("Monitoring dashboard will run with limited functionality")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and configure the monitoring dashboard
    dashboard = MonitoringDashboard(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator,
        result_aggregator=result_aggregator,
        output_dir=args.output_dir
    )
    
    # Configure dashboard
    dashboard.configure({
        "theme": args.theme,
        "auto_refresh": args.refresh,
        "enable_alerts": not args.no_alerts,
        "real_time_enabled": args.real_time or False
    })
    
    print(f"Dashboard configured with theme: {args.theme}, refresh: {args.refresh}s")
    
    # Auto-open in browser if requested
    if args.browser:
        try:
            import webbrowser
            url = f"http://{args.host}:{args.port}"
            print(f"Opening dashboard in browser: {url}")
            webbrowser.open(url)
        except Exception as e:
            print(f"Warning: Failed to open browser: {e}")
    
    try:
        # Start dashboard (this will block until interrupted)
        print(f"Starting monitoring dashboard at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server")
        dashboard.start()
    except KeyboardInterrupt:
        print("\nStopping monitoring dashboard...")
        dashboard.stop()
    except Exception as e:
        print(f"Error running monitoring dashboard: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        dashboard.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()