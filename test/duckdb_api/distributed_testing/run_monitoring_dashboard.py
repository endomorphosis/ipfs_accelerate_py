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
    --time-range DAYS       Time range in days for result aggregation (default: 7)
    --disable-aggregator    Disable result aggregator integration
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

# Import result aggregator integration if available
try:
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_result_aggregator_integration import ResultAggregatorIntegration
    RESULT_AGGREGATOR_INTEGRATION_AVAILABLE = True
except ImportError:
    RESULT_AGGREGATOR_INTEGRATION_AVAILABLE = False

# Import E2E test integration if available
try:
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_e2e_integration import E2ETestResultsIntegration
    E2E_TEST_INTEGRATION_AVAILABLE = True
except ImportError:
    E2E_TEST_INTEGRATION_AVAILABLE = False

# Import Advanced Visualization System integration if available
try:
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import VisualizationDashboardIntegration
    VISUALIZATION_INTEGRATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_INTEGRATION_AVAILABLE = False

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
    parser.add_argument("--time-range", type=int, default=7,
                       help="Time range in days for result aggregation")
    parser.add_argument("--disable-aggregator", action="store_true",
                       help="Disable result aggregator integration")
    
    # E2E test integration options
    parser.add_argument("--enable-e2e-test-integration", action="store_true",
                       help="Enable integration with E2E testing framework")
    parser.add_argument("--e2e-report-dir", default="./e2e_test_reports",
                       help="Directory for E2E test reports")
    parser.add_argument("--e2e-visualization-dir", default="./e2e_visualizations",
                       help="Directory for E2E test visualizations")
    
    # Advanced Visualization System integration options
    parser.add_argument("--enable-visualization-integration", action="store_true",
                       help="Enable integration with Advanced Visualization System")
    parser.add_argument("--dashboard-dir", default="./dashboards",
                       help="Directory to store visualization dashboards")
    
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
    
    # Create database manager if available
    db_manager = None
    if DB_MANAGER_AVAILABLE:
        try:
            db_path = args.db_path or "./benchmark_db.duckdb"
            db_manager = BenchmarkDBManager(db_path)
            print(f"Using database at: {db_path}")
        except Exception as e:
            print(f"Warning: Failed to initialize database manager: {e}")
            print("Some dashboard features will be limited")
    
    # Create result aggregator if available
    result_aggregator = None
    if RESULT_AGGREGATOR_AVAILABLE and not args.disable_aggregator:
        try:
            result_aggregator = ResultAggregatorService(db_manager=db_manager)
            print("Result aggregator initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize result aggregator: {e}")
            print("Dashboard will run with limited result aggregation functionality")
    
    # Create enhanced result aggregator integration if available
    result_aggregator_integration = None
    if RESULT_AGGREGATOR_INTEGRATION_AVAILABLE and result_aggregator and not args.disable_aggregator:
        try:
            result_aggregator_integration = ResultAggregatorIntegration(
                result_aggregator=result_aggregator,
                output_dir=args.output_dir
            )
            # Configure integration
            result_aggregator_integration.configure({
                "theme": args.theme,
                "max_items_in_charts": 10,
                "chart_height": 500,
                "chart_width": 900,
                "enable_annotations": True
            })
            print("Enhanced result aggregator integration initialized successfully")
            
            # Generate initial dashboard summary
            print(f"Generating initial dashboard summary for the last {args.time_range} days...")
            dashboard_summary = result_aggregator_integration.create_dashboard_result_summary(args.time_range)
            if "error" not in dashboard_summary:
                print("Initial dashboard summary generated successfully")
                if args.debug:
                    # Print summary statistics in debug mode
                    stats = dashboard_summary.get("overall_stats", {})
                    print(f"Total tests run: {stats.get('total_tests_run', 0)}")
                    print(f"Model-hardware pairs: {stats.get('total_model_hardware_pairs', 0)}")
                    print(f"Compatibility rate: {stats.get('compatibility_rate', 0):.1f}%")
                    print(f"Integration test pass rate: {stats.get('integration_pass_rate', 0):.1f}%")
                    print(f"Web platform success rate: {stats.get('web_platform_success_rate', 0):.1f}%")
            else:
                print(f"Warning: Failed to generate initial dashboard summary: {dashboard_summary['error']}")
        except Exception as e:
            print(f"Warning: Failed to initialize enhanced result aggregator integration: {e}")
            print("Advanced result visualization will be limited")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and configure the monitoring dashboard
    dashboard = MonitoringDashboard(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator,
        result_aggregator_url=None,  # Will be set up separately
        refresh_interval=args.refresh,
        theme=args.theme,
        enable_result_aggregator_integration=not args.disable_aggregator and result_aggregator_integration is not None,
        result_aggregator_integration=result_aggregator_integration,
        enable_e2e_test_integration=args.enable_e2e_test_integration,
        e2e_test_integration=None,  # Will be set up separately
        enable_performance_analytics=True,
        enable_visualization_integration=args.enable_visualization_integration,
        visualization_integration=None,  # Will be set up automatically
        dashboard_dir=args.dashboard_dir
    )
    
    # Dashboard is already configured through constructor parameters
    
    # Store result aggregator integration in dashboard for access in templates
    if result_aggregator_integration:
        dashboard.result_aggregator_integration = result_aggregator_integration
        dashboard.result_aggregator_time_range = args.time_range
    
    # Initialize E2E test integration if enabled
    e2e_test_integration = None
    if args.enable_e2e_test_integration and E2E_TEST_INTEGRATION_AVAILABLE:
        try:
            # Create directories if needed
            os.makedirs(args.e2e_report_dir, exist_ok=True)
            os.makedirs(args.e2e_visualization_dir, exist_ok=True)
            
            # Create E2E test integration
            e2e_test_integration = E2ETestResultsIntegration(
                report_dir=args.e2e_report_dir,
                visualization_dir=args.e2e_visualization_dir
            )
            
            # Store in dashboard
            dashboard.e2e_test_integration = e2e_test_integration
            dashboard.enable_e2e_test_integration = True
            
            print(f"E2E test integration initialized with report dir: {args.e2e_report_dir}")
        except Exception as e:
            print(f"Warning: Failed to initialize E2E test integration: {e}")
            dashboard.enable_e2e_test_integration = False
            if args.debug:
                import traceback
                traceback.print_exc()
    else:
        dashboard.enable_e2e_test_integration = False

    # Initialize Visualization Dashboard integration if enabled
    visualization_integration = None
    if args.enable_visualization_integration and VISUALIZATION_INTEGRATION_AVAILABLE:
        try:
            # Create dashboard directory if needed
            os.makedirs(args.dashboard_dir, exist_ok=True)
            
            # Create symbolic link to dashboards in output directory for serving
            dashboards_link = os.path.join(args.output_dir, 'dashboards')
            if not os.path.exists(dashboards_link):
                try:
                    # Use relative path for the link target if possible
                    target_path = os.path.relpath(args.dashboard_dir, os.path.dirname(dashboards_link))
                    os.symlink(target_path, dashboards_link, target_is_directory=True)
                except Exception as e:
                    print(f"Warning: Failed to create symbolic link to dashboards: {e}")
                    # Fall back to normal directory
                    os.makedirs(dashboards_link, exist_ok=True)
            
            # Create visualization integration
            visualization_integration = VisualizationDashboardIntegration(
                dashboard_dir=args.dashboard_dir,
                integration_dir=os.path.join(args.dashboard_dir, 'monitor_integration')
            )
            
            # Store in dashboard
            dashboard.visualization_integration = visualization_integration
            dashboard.enable_visualization_integration = True
            
            print(f"Advanced Visualization System integration initialized with dashboard dir: {args.dashboard_dir}")
            
            # Create default dashboards for main pages if they don't exist
            if visualization_integration.visualization_available:
                # Check if we already have dashboards for main pages
                existing_dashboards = visualization_integration.embedded_dashboards
                
                # Create dashboard for Overview page if it doesn't exist
                if not any(dash.get('page') == 'index' for dash in existing_dashboards.values()):
                    print("Creating default Overview dashboard")
                    visualization_integration.create_embedded_dashboard(
                        name="overview_dashboard",
                        page="index",
                        template="overview",
                        title="System Overview Dashboard",
                        description="Overview of system performance metrics",
                        position="below"
                    )
                
                # Create dashboard for Results page if it doesn't exist
                if not any(dash.get('page') == 'results' for dash in existing_dashboards.values()):
                    print("Creating default Results dashboard")
                    visualization_integration.create_embedded_dashboard(
                        name="results_dashboard",
                        page="results",
                        template="model_analysis",
                        title="Model Performance Dashboard",
                        description="Detailed analysis of model performance metrics",
                        position="below"
                    )
                
                # Create dashboard for Performance Analytics page if it doesn't exist
                if not any(dash.get('page') == 'performance-analytics' for dash in existing_dashboards.values()):
                    print("Creating default Performance Analytics dashboard")
                    visualization_integration.create_embedded_dashboard(
                        name="performance_analytics_dashboard",
                        page="performance-analytics",
                        template="hardware_comparison",
                        title="Hardware Comparison Dashboard",
                        description="Detailed comparison of hardware performance metrics",
                        position="below"
                    )
                
        except Exception as e:
            print(f"Warning: Failed to initialize Visualization Dashboard integration: {e}")
            dashboard.enable_visualization_integration = False
            if args.debug:
                import traceback
                traceback.print_exc()
    else:
        dashboard.enable_visualization_integration = False
    
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
        import asyncio
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        print("\nStopping monitoring dashboard...")
    except Exception as e:
        print(f"Error running monitoring dashboard: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()