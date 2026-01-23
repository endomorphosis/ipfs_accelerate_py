#!/usr/bin/env python3
"""
Dashboard Example for the Distributed Testing Framework.

This example demonstrates how to set up and use the result aggregator service
and dashboard components to visualize test results.

Usage:
  python dashboard_example.py --host localhost --port 8081 --output ./dashboards

"""

import os
import sys
import argparse
import anyio
import threading
import webbrowser
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import dashboard components
from duckdb_api.distributed_testing.dashboard.dashboard_generator import DashboardGenerator
from duckdb_api.distributed_testing.dashboard.dashboard_server import DashboardServer
from duckdb_api.distributed_testing.dashboard.visualization import VisualizationEngine

# Import result aggregator
from duckdb_api.distributed_testing.result_aggregator.service import ResultAggregatorService

# Import database manager
from duckdb_api.core.db_manager import BenchmarkDBManager


def generate_sample_dashboard(result_aggregator, output_dir):
    """
    Generate a sample dashboard using the result aggregator.
    
    Args:
        result_aggregator: Result aggregator service
        output_dir: Directory to save the dashboard
    
    Returns:
        Path to the generated dashboard
    """
    # Create dashboard generator
    dashboard_generator = DashboardGenerator(
        result_aggregator=result_aggregator,
        output_dir=output_dir
    )
    
    # Configure dashboard generator
    dashboard_generator.configure({
        "theme": "light",
        "refresh_interval": 0,
        "include_performance_charts": True,
        "include_regression_detection": True,
        "include_dimension_analysis": True,
        "include_test_details": True,
        "include_worker_details": True,
        "max_items_per_section": 10
    })
    
    # Generate dashboard
    dashboard_path = dashboard_generator.generate_dashboard()
    print(f"Generated dashboard: {dashboard_path}")
    
    # Generate regression report
    regression_path = dashboard_generator.generate_report("regression")
    print(f"Generated regression report: {regression_path}")
    
    return dashboard_path


def start_dashboard_server(result_aggregator, host, port, output_dir):
    """
    Start the dashboard server.
    
    Args:
        result_aggregator: Result aggregator service
        host: Host to bind the server to
        port: Port to bind the server to
        output_dir: Directory to save dashboards
    
    Returns:
        Dashboard server thread
    """
    # Create dashboard server
    server = DashboardServer(
        host=host,
        port=port,
        result_aggregator=result_aggregator,
        output_dir=output_dir
    )
    
    # Configure server
    server.configure({
        "auto_refresh": 60,
        "theme": "light",
        "max_items_per_page": 50,
        "default_report_type": "performance",
        "api_cache_time": 10
    })
    
    # Start server in a separate thread
    server_thread = server.start_async()
    print(f"Dashboard server started at http://{host}:{port}")
    
    return server_thread


def create_visualizations(result_aggregator, output_dir):
    """
    Create sample visualizations using the visualization engine.
    
    Args:
        result_aggregator: Result aggregator service
        output_dir: Directory to save visualizations
    """
    # Create visualization engine
    viz_engine = VisualizationEngine(
        result_aggregator=result_aggregator,
        output_dir=os.path.join(output_dir, "visualizations")
    )
    
    # Configure visualization engine
    viz_engine.configure({
        "theme": "light",
        "interactive": True,
        "static_format": "png",
        "width": 1200,
        "height": 800,
        "dpi": 100,
        "include_annotations": True
    })
    
    # Get performance data for visualization
    results = result_aggregator.aggregate_results(
        result_type="performance",
        aggregation_level="hardware"
    )
    
    # Create time series visualization
    if "results" in results and "basic_statistics" in results["results"]:
        time_series_data = {}
        
        # Extract time series data from the database
        for hardware_id, stats in results["results"]["basic_statistics"].items():
            if "throughput_items_per_second" in stats:
                # Get historical performance data
                historical_results = result_aggregator.get_comparison_report(
                    result_type="performance",
                    aggregation_level="hardware",
                    filter_params={"hardware_id": hardware_id}
                )
                
                # Create time series for this hardware
                if "comparisons" in historical_results:
                    for comparison in historical_results["comparisons"]:
                        if comparison["group"] == hardware_id and comparison["metric"] == "throughput_items_per_second":
                            current = comparison["current_mean"]
                            historical = comparison["historical_mean"]
                            
                            # Create simple time series (current and historical)
                            from datetime import datetime, timedelta
                            time_series_data[hardware_id] = [
                                (datetime.now() - timedelta(days=7), historical),
                                (datetime.now(), current)
                            ]
        
        # Create visualization if we have data
        if time_series_data:
            viz_path = viz_engine.create_visualization(
                "time_series",
                {
                    "time_series": time_series_data,
                    "metric": "throughput_items_per_second",
                    "title": "Throughput by Hardware"
                }
            )
            print(f"Created time series visualization: {viz_path}")
    
    # Create dimension comparison visualization
    dimension_results = result_aggregator.aggregate_results(
        result_type="performance",
        aggregation_level="model"
    )
    
    if "results" in dimension_results and "basic_statistics" in dimension_results["results"]:
        # Extract dimension values
        dimension_values = {}
        
        for model_id, stats in dimension_results["results"]["basic_statistics"].items():
            if "throughput_items_per_second" in stats:
                dimension_values[model_id] = stats["throughput_items_per_second"]["mean"]
        
        # Create visualization if we have data
        if dimension_values:
            viz_path = viz_engine.create_visualization(
                "dimension_comparison",
                {
                    "dimension": "model",
                    "metric": "throughput_items_per_second",
                    "values": dimension_values,
                    "title": "Throughput by Model"
                }
            )
            print(f"Created dimension comparison visualization: {viz_path}")
    
    # Create regression visualization
    regression_results = result_aggregator.get_result_anomalies(
        result_type="performance",
        aggregation_level="model_hardware"
    )
    
    if regression_results["anomaly_count"] > 0:
        viz_path = viz_engine.create_visualization(
            "regression_analysis",
            {
                "regressions": regression_results["anomalies"],
                "title": "Performance Regressions"
            }
        )
        print(f"Created regression visualization: {viz_path}")


def main():
    """Main function to run the dashboard example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dashboard Example for Distributed Testing Framework")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind the server to")
    parser.add_argument("--output-dir", default="./dashboards", help="Directory to save dashboards")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb", help="Path to the database file")
    parser.add_argument("--auto-open", action="store_true", help="Automatically open dashboard in browser")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create database manager
        if os.path.exists(args.db_path):
            print(f"Using existing database: {args.db_path}")
            db_manager = BenchmarkDBManager(db_path=args.db_path)
        else:
            print(f"Creating new database: {args.db_path}")
            db_manager = BenchmarkDBManager(db_path=args.db_path, create_if_not_exists=True)
        
        # Create result aggregator
        result_aggregator = ResultAggregatorService(db_manager=db_manager)
        
        # Configure result aggregator
        result_aggregator.configure({
            "cache_ttl_seconds": 300,
            "anomaly_threshold": 2.5,
            "comparative_lookback_days": 30,
            "normalize_metrics": True,
            "deduplication_enabled": True,
            "model_family_grouping": True
        })
        
        # Generate sample dashboard
        dashboard_path = generate_sample_dashboard(result_aggregator, args.output_dir)
        
        # Create sample visualizations
        create_visualizations(result_aggregator, args.output_dir)
        
        # Start dashboard server
        server_thread = start_dashboard_server(
            result_aggregator, args.host, args.port, args.output_dir
        )
        
        # Open dashboard in browser if requested
        if args.auto_open:
            # Wait a moment for the server to start
            import time
            time.sleep(1)
            
            # Open the dashboard URL
            dashboard_url = f"http://{args.host}:{args.port}/dashboard"
            print(f"Opening dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        
        # Keep the main thread running
        try:
            while True:
                # Sleep to avoid high CPU usage
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()