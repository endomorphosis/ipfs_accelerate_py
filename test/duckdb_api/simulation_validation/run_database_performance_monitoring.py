#!/usr/bin/env python3
"""
Command-line tool for Database Performance Monitoring in the Simulation Accuracy and Validation Framework.

This script provides a command-line interface for monitoring, optimizing, and visualizing
database performance metrics. It can be used to create dashboards, run optimizations, and
perform maintenance tasks on the DuckDB database used by the Simulation Accuracy and Validation
Framework.

Enhanced with comprehensive dashboard integration and automation features for real-time
monitoring and visualization of database performance metrics. Now includes automated 
optimization capabilities through the AutomatedOptimizationManager component.
"""

import os
import sys
import argparse
import logging
import json
import datetime
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_performance_monitoring")

# Import required components
try:
    from data.duckdb.simulation_validation.db_performance_optimizer import DBPerformanceOptimizer
    from data.duckdb.simulation_validation.visualization.monitoring_dashboard_connector import MonitoringDashboardConnector
    from data.duckdb.simulation_validation.automated_optimization_manager import (
        AutomatedOptimizationManager, 
        get_optimization_manager
    )
    from data.duckdb.simulation_validation.database_predictive_analytics import DatabasePredictiveAnalytics
except ImportError:
    logger.error("Failed to import required modules. Make sure duckdb_api is properly installed.")
    sys.exit(1)


def get_optimizer(db_path: str, enable_caching: bool = True) -> DBPerformanceOptimizer:
    """
    Get a configured DBPerformanceOptimizer instance.
    
    Args:
        db_path: Path to the DuckDB database
        enable_caching: Whether to enable query caching
        
    Returns:
        Configured DBPerformanceOptimizer instance
    """
    try:
        optimizer = DBPerformanceOptimizer(
            db_path=db_path,
            enable_caching=enable_caching,
            cache_size=100,
            cache_ttl=300
        )
        return optimizer
    except Exception as e:
        logger.error(f"Failed to initialize database performance optimizer: {e}")
        sys.exit(1)


def get_dashboard_connector(
    dashboard_url: str,
    api_key: str,
    optimizer: DBPerformanceOptimizer
) -> MonitoringDashboardConnector:
    """
    Get a configured MonitoringDashboardConnector instance.
    
    Args:
        dashboard_url: URL of the monitoring dashboard API
        api_key: API key for the dashboard
        optimizer: DBPerformanceOptimizer instance
        
    Returns:
        Configured MonitoringDashboardConnector instance
    """
    try:
        connector = MonitoringDashboardConnector(
            dashboard_url=dashboard_url,
            dashboard_api_key=api_key,
            db_optimizer=optimizer
        )
        return connector
    except Exception as e:
        logger.error(f"Failed to initialize dashboard connector: {e}")
        sys.exit(1)


def get_predictive_analytics(
    auto_manager: AutomatedOptimizationManager,
    config_file: Optional[str] = None
) -> DatabasePredictiveAnalytics:
    """
    Get a configured DatabasePredictiveAnalytics instance.
    
    Args:
        auto_manager: AutomatedOptimizationManager instance
        config_file: Path to configuration file for predictive analytics
        
    Returns:
        Configured DatabasePredictiveAnalytics instance
    """
    try:
        # Load configuration from file if provided
        config = None
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded predictive analytics configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading predictive analytics configuration file: {e}")
        
        # Create predictive analytics instance
        predictive = DatabasePredictiveAnalytics(
            automated_optimization_manager=auto_manager,
            config=config
        )
        return predictive
    except Exception as e:
        logger.error(f"Failed to initialize predictive analytics: {e}")
        sys.exit(1)


def show_metrics(optimizer: DBPerformanceOptimizer, format_output: str = "text", output_file: Optional[str] = None) -> None:
    """
    Show database performance metrics.
    
    Args:
        optimizer: DBPerformanceOptimizer instance
        format_output: Output format (text, json, markdown, html)
        output_file: Optional file path to write output to
    """
    try:
        # Get performance metrics
        metrics = optimizer.get_performance_metrics()
        overall_status = optimizer.get_overall_status()
        
        # Prepare output
        if format_output.lower() == "json":
            # JSON output
            output = {
                "metrics": metrics,
                "overall_status": overall_status,
                "timestamp": datetime.datetime.now().isoformat()
            }
            formatted_output = json.dumps(output, indent=2)
        
        elif format_output.lower() == "markdown":
            # Markdown output
            formatted_output = f"# Database Performance Metrics\n\n"
            formatted_output += f"**Overall Status:** {overall_status.upper()}\n\n"
            formatted_output += f"**Timestamp:** {datetime.datetime.now().isoformat()}\n\n"
            
            # Create markdown table for metrics
            formatted_output += "| Metric | Value | Status | Change |\n"
            formatted_output += "|--------|-------|--------|--------|\n"
            
            for metric_name, metric_data in metrics.items():
                value = metric_data.get("value")
                status = metric_data.get("status", "unknown")
                unit = metric_data.get("unit", "")
                
                # Format value for display
                if metric_name == "storage_size" and value > 1000000:
                    formatted_value = f"{value / 1000000:.2f} MB"
                elif metric_name in ["index_efficiency", "vacuum_status", "cache_performance"]:
                    formatted_value = f"{value:.1f}%"
                elif metric_name == "compression_ratio":
                    formatted_value = f"{value:.2f}x"
                else:
                    formatted_value = f"{value:.1f} {unit}"
                
                # Add status indicator
                status_indicator = {
                    "good": "✅",
                    "warning": "⚠️",
                    "error": "❌",
                    "unknown": "❓"
                }.get(status, "❓")
                
                # Add change info
                change_info = ""
                if "change_pct" in metric_data:
                    change_pct = metric_data["change_pct"]
                    direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
                    change_info = f"{direction} {abs(change_pct):.1f}%"
                
                # Add row to table
                formatted_output += f"| {metric_name.replace('_', ' ').title()} | {formatted_value} | {status_indicator} {status.title()} | {change_info} |\n"
            
            # Add detailed metrics section
            formatted_output += "\n## Detailed Metrics\n\n"
            for metric_name, metric_data in metrics.items():
                formatted_output += f"### {metric_name.replace('_', ' ').title()}\n\n"
                for key, value in metric_data.items():
                    if key not in ["value", "status", "unit", "history", "timestamp"]:
                        formatted_output += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                formatted_output += "\n"

        elif format_output.lower() == "html":
            # HTML output
            formatted_output = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Database Performance Metrics</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .status-good {{ color: green; }}
                    .status-warning {{ color: orange; }}
                    .status-error {{ color: red; }}
                    .status-unknown {{ color: gray; }}
                    .metric-table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .metric-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .metric-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }}
                    .change-up {{ color: {{"green" if overall_status == "good" else "red"}}; }}
                    .change-down {{ color: {{"red" if overall_status == "good" else "green"}}; }}
                    .metric-details {{ margin-top: 10px; padding: 10px; background-color: #f8f8f8; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Database Performance Metrics</h1>
                <p><strong>Overall Status:</strong> <span class="status-{overall_status}">{overall_status.upper()}</span></p>
                <p><strong>Timestamp:</strong> {datetime.datetime.now().isoformat()}</p>
                
                <table class="metric-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                        <th>Change</th>
                    </tr>
            """
            
            for metric_name, metric_data in metrics.items():
                value = metric_data.get("value")
                status = metric_data.get("status", "unknown")
                unit = metric_data.get("unit", "")
                
                # Format value for display
                if metric_name == "storage_size" and value > 1000000:
                    formatted_value = f"{value / 1000000:.2f} MB"
                elif metric_name in ["index_efficiency", "vacuum_status", "cache_performance"]:
                    formatted_value = f"{value:.1f}%"
                elif metric_name == "compression_ratio":
                    formatted_value = f"{value:.2f}x"
                else:
                    formatted_value = f"{value:.1f} {unit}"
                
                # Add status indicator
                status_indicator = {
                    "good": "✅",
                    "warning": "⚠️",
                    "error": "❌",
                    "unknown": "❓"
                }.get(status, "❓")
                
                # Add change info
                change_info = ""
                change_class = ""
                if "change_pct" in metric_data:
                    change_pct = metric_data["change_pct"]
                    if change_pct > 0:
                        direction = "↑"
                        change_class = "change-up"
                    elif change_pct < 0:
                        direction = "↓"
                        change_class = "change-down"
                    else:
                        direction = "→"
                        change_class = ""
                    change_info = f"{direction} {abs(change_pct):.1f}%"
                
                formatted_output += f"""
                    <tr>
                        <td>{metric_name.replace('_', ' ').title()}</td>
                        <td>{formatted_value}</td>
                        <td class="status-{status}">{status_indicator} {status.title()}</td>
                        <td class="{change_class}">{change_info}</td>
                    </tr>
                """
            
            # Close the table
            formatted_output += """
                </table>
                
                <h2>Detailed Metrics</h2>
            """
            
            # Add detailed metrics
            for metric_name, metric_data in metrics.items():
                formatted_output += f"""
                <div class="metric-details">
                    <h3>{metric_name.replace('_', ' ').title()}</h3>
                    <ul>
                """
                
                for key, value in metric_data.items():
                    if key not in ["value", "status", "unit", "history", "timestamp"]:
                        formatted_output += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>\n"
                
                formatted_output += """
                    </ul>
                </div>
                """
            
            # Close HTML
            formatted_output += """
            </body>
            </html>
            """
        
        else:
            # Default text output
            formatted_output = f"\n=== Database Performance Metrics ===\n"
            formatted_output += f"Overall Status: {overall_status.upper()}\n"
            formatted_output += f"Timestamp: {datetime.datetime.now().isoformat()}\n\n"
            
            for metric_name, metric_data in metrics.items():
                value = metric_data.get("value")
                status = metric_data.get("status", "unknown")
                unit = metric_data.get("unit", "")
                
                # Format value for display
                if metric_name == "storage_size" and value > 1000000:
                    formatted_value = f"{value / 1000000:.2f} MB"
                elif metric_name in ["index_efficiency", "vacuum_status", "cache_performance"]:
                    formatted_value = f"{value:.1f}%"
                elif metric_name == "compression_ratio":
                    formatted_value = f"{value:.2f}x"
                else:
                    formatted_value = f"{value:.1f} {unit}"
                
                # Add status indicator
                status_indicator = {
                    "good": "✅",
                    "warning": "⚠️",
                    "error": "❌",
                    "unknown": "❓"
                }.get(status, "❓")
                
                formatted_output += f"{status_indicator} {metric_name.replace('_', ' ').title()}: {formatted_value}\n"
                
                # Show change percentage if available
                if "change_pct" in metric_data:
                    change_pct = metric_data["change_pct"]
                    direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
                    formatted_output += f"   {direction} {abs(change_pct):.1f}% from previous measurement\n"
                
                formatted_output += "\n"
        
        # Output to file or print to console
        if output_file:
            with open(output_file, 'w') as f:
                f.write(formatted_output)
            print(f"Metrics saved to {output_file}")
        else:
            print(formatted_output)
            
    except Exception as e:
        logger.error(f"Error showing metrics: {e}")
        sys.exit(1)


def optimize_database(optimizer: DBPerformanceOptimizer, show_before_after: bool = False) -> None:
    """
    Run database optimization.
    
    Args:
        optimizer: DBPerformanceOptimizer instance
        show_before_after: Whether to show before/after comparison
    """
    try:
        print("\n=== Database Optimization ===")
        
        # Get metrics before optimization if requested
        before_metrics = None
        if show_before_after:
            before_metrics = optimizer.get_performance_metrics()
            print("\nBefore optimization metrics:")
            for metric_name in ["query_time", "storage_size", "index_efficiency", "vacuum_status"]:
                if metric_name in before_metrics:
                    value = before_metrics[metric_name].get("value")
                    unit = before_metrics[metric_name].get("unit", "")
                    formatted_value = value
                    if metric_name in ["index_efficiency", "vacuum_status"]:
                        formatted_value = f"{value:.1f}%"
                    print(f"- {metric_name.replace('_', ' ').title()}: {formatted_value} {unit}")
            print("")
        
        print("Creating indexes...")
        optimizer.create_indexes()
        
        print("Analyzing tables...")
        optimizer.analyze_tables()
        
        print("Running vacuum...")
        result = optimizer.optimize_database()
        
        if result:
            print("\n✅ Database optimization completed successfully")
            
            # Show updated metrics
            print("\nUpdated performance metrics:")
            after_metrics = optimizer.get_performance_metrics()
            
            for metric_name in ["query_time", "index_efficiency", "vacuum_status"]:
                if metric_name in after_metrics:
                    value = after_metrics[metric_name].get("value")
                    unit = after_metrics[metric_name].get("unit", "")
                    formatted_value = value
                    if metric_name in ["index_efficiency", "vacuum_status"]:
                        formatted_value = f"{value:.1f}%"
                        
                    # Calculate improvement if before metrics are available
                    improvement = ""
                    if show_before_after and before_metrics and metric_name in before_metrics:
                        before_value = before_metrics[metric_name].get("value")
                        if before_value is not None and value is not None:
                            # For metrics where higher is better
                            if metric_name in ["index_efficiency", "vacuum_status"]:
                                change = value - before_value
                                change_pct = (change / before_value * 100) if before_value != 0 else 0
                                if change > 0:
                                    improvement = f" (improved by {change_pct:.1f}%)"
                                elif change < 0:
                                    improvement = f" (decreased by {abs(change_pct):.1f}%)"
                            # For metrics where lower is better
                            elif metric_name == "query_time":
                                change = before_value - value
                                change_pct = (change / before_value * 100) if before_value != 0 else 0
                                if change > 0:
                                    improvement = f" (improved by {change_pct:.1f}%)"
                                elif change < 0:
                                    improvement = f" (slowed by {abs(change_pct):.1f}%)"
                    
                    print(f"- {metric_name.replace('_', ' ').title()}: {formatted_value} {unit}{improvement}")
        else:
            print("\n❌ Database optimization failed")
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        sys.exit(1)


def create_dashboard(
    connector: MonitoringDashboardConnector,
    dashboard_title: str,
    visualization_style: str,
    create_alerts: bool,
    auto_update: bool,
    metrics: Optional[List[str]] = None,
    update_interval: int = 3600
) -> None:
    """
    Create a database performance dashboard.
    
    Args:
        connector: MonitoringDashboardConnector instance
        dashboard_title: Title for the dashboard
        visualization_style: Style of visualization (detailed, compact, overview)
        create_alerts: Whether to create alerts for critical metrics
        auto_update: Whether to set up scheduled updates
        metrics: List of specific metrics to include (None for all available)
        update_interval: Interval for automatic updates in seconds
    """
    try:
        print("\n=== Creating Database Performance Dashboard ===")
        
        result = connector.create_database_performance_dashboard(
            dashboard_title=dashboard_title,
            metrics=metrics,
            visualization_style=visualization_style,
            create_alerts=create_alerts,
            auto_update=auto_update,
            update_interval=update_interval,
            refresh_interval=min(300, update_interval // 2)  # Panel refresh interval (5 min or half update interval)
        )
        
        if result["success"]:
            print(f"\n✅ Dashboard created successfully")
            print(f"Dashboard URL: {result['dashboard_url']}")
            print(f"Dashboard ID: {result['dashboard_id']}")
            print(f"Created {result['panels_created']} panels and {result['alerts_created']} alerts")
            print(f"Visualization style: {visualization_style}")
            
            # Show metrics being monitored
            if "metrics_monitored" in result and result["metrics_monitored"]:
                print("\nMonitored metrics:")
                for metric in result["metrics_monitored"]:
                    print(f"- {metric.replace('_', ' ').title()}")
            
            if auto_update:
                print(f"\nAuto-updates enabled with interval: {result['update_interval']} seconds")
                print(f"Panel refresh interval: {result.get('refresh_interval', 300)} seconds")
        else:
            print(f"\n❌ Failed to create dashboard: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        sys.exit(1)


def create_complete_solution(
    connector: MonitoringDashboardConnector,
    dashboard_title: str,
    hardware_types: List[str],
    model_types: List[str],
    visualization_style: str,
    performance_metrics: Optional[List[str]] = None,
    database_metrics: Optional[List[str]] = None,
    refresh_interval: int = 300
) -> None:
    """
    Create a complete monitoring solution.
    
    Args:
        connector: MonitoringDashboardConnector instance
        dashboard_title: Title for the dashboard
        hardware_types: List of hardware types to monitor
        model_types: List of model types to monitor
        visualization_style: Style of visualization (detailed, compact, overview)
        performance_metrics: List of performance metrics to monitor
        database_metrics: List of database metrics to monitor
        refresh_interval: Refresh interval in seconds
    """
    try:
        print("\n=== Creating Complete Monitoring Solution ===")
        
        # Define default performance metrics if not provided
        if not performance_metrics:
            performance_metrics = [
                "throughput_items_per_second", 
                "average_latency_ms", 
                "memory_peak_mb", 
                "power_consumption_w"
            ]
        
        result = connector.create_complete_monitoring_solution(
            dashboard_title=dashboard_title,
            include_database_performance=True,
            include_validation_metrics=True,
            hardware_types=hardware_types,
            model_types=model_types,
            performance_metrics=performance_metrics,
            database_metrics=database_metrics,
            refresh_interval=refresh_interval,
            visualization_style=visualization_style,
            create_alerts=True
        )
        
        if result["success"]:
            print(f"\n✅ Complete monitoring solution created successfully")
            print(f"Dashboard URL: {result['dashboard_url']}")
            print(f"Dashboard ID: {result['dashboard_id']}")
            print(f"Component count: {result['component_count']}")
            print(f"Visualization style: {visualization_style}")
            print(f"Refresh interval: {refresh_interval} seconds")
            
            # Show hardware and model types
            print("\nHardware types monitored:")
            for hw_type in hardware_types:
                print(f"- {hw_type}")
            
            print("\nModel types monitored:")
            for model_type in model_types:
                print(f"- {model_type}")
            
            print("\nIncludes database metrics: Yes")
            print("Includes validation metrics: Yes")
        else:
            print(f"\n❌ Failed to create monitoring solution: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error creating complete monitoring solution: {e}")
        sys.exit(1)


def update_dashboard(
    connector: MonitoringDashboardConnector,
    dashboard_id: str
) -> None:
    """
    Update an existing dashboard with latest metrics.
    
    Args:
        connector: MonitoringDashboardConnector instance
        dashboard_id: ID of the dashboard to update
    """
    try:
        print("\n=== Updating Dashboard Metrics ===")
        
        result = connector.update_database_performance_metrics(
            dashboard_id=dashboard_id,
            include_history=True,
            format_values=True
        )
        
        if result["success"]:
            print(f"\n✅ Dashboard updated successfully")
            print(f"Updated {result['updated_panels']} panels")
            
            # Show which metrics were updated
            if "metrics_updated" in result and result["metrics_updated"]:
                print("\nUpdated metrics:")
                for metric in result["metrics_updated"]:
                    print(f"- {metric.replace('_', ' ').title()}")
            
            print(f"\nOverall status: {result.get('overall_status', 'unknown').upper()}")
            print(f"Timestamp: {result.get('timestamp', datetime.datetime.now().isoformat())}")
        else:
            print(f"\n❌ Failed to update dashboard: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        sys.exit(1)


def continuous_monitoring(
    connector: MonitoringDashboardConnector,
    dashboard_id: str,
    interval: int,
    duration: Optional[int] = None,
    console_output: bool = True
) -> None:
    """
    Continuously monitor database performance and update dashboard.
    
    Args:
        connector: MonitoringDashboardConnector instance
        dashboard_id: ID of the dashboard to update
        interval: Update interval in seconds
        duration: Optional duration in seconds (None for indefinite)
        console_output: Whether to print metrics to console
    """
    try:
        print(f"\n=== Starting Continuous Monitoring (Update every {interval}s) ===")
        if duration:
            print(f"Monitoring will run for {duration} seconds")
            end_time = time.time() + duration
        else:
            print("Monitoring will run until interrupted (Ctrl+C)")
            end_time = None
        
        # Run until duration expires or interrupted
        iteration = 1
        try:
            while True:
                print(f"\nIteration {iteration} - {datetime.datetime.now().isoformat()}")
                
                # Get current metrics
                metrics = connector.db_optimizer.get_performance_metrics()
                overall_status = connector.db_optimizer.get_overall_status()
                
                # Display metrics in console if requested
                if console_output:
                    print(f"Overall Status: {overall_status.upper()}")
                    for metric_name, metric_data in metrics.items():
                        value = metric_data.get("value")
                        status = metric_data.get("status", "unknown")
                        unit = metric_data.get("unit", "")
                        
                        # Format value for display
                        if metric_name == "storage_size" and value > 1000000:
                            formatted_value = f"{value / 1000000:.2f} MB"
                        elif metric_name in ["index_efficiency", "vacuum_status", "cache_performance"]:
                            formatted_value = f"{value:.1f}%"
                        elif metric_name == "compression_ratio":
                            formatted_value = f"{value:.2f}x"
                        else:
                            formatted_value = f"{value:.1f} {unit}"
                        
                        # Add status indicator
                        status_indicator = {
                            "good": "✅",
                            "warning": "⚠️",
                            "error": "❌",
                            "unknown": "❓"
                        }.get(status, "❓")
                        
                        print(f"{status_indicator} {metric_name.replace('_', ' ').title()}: {formatted_value}")
                
                # Update dashboard
                result = connector.update_database_performance_metrics(
                    dashboard_id=dashboard_id,
                    include_history=True,
                    format_values=True
                )
                
                if result["success"]:
                    print(f"Updated {result['updated_panels']} dashboard panels")
                else:
                    print(f"Failed to update dashboard: {result.get('error', 'Unknown error')}")
                
                # Check if duration has expired
                if duration and time.time() >= end_time:
                    print(f"\nMonitoring duration of {duration} seconds completed")
                    break
                
                # Sleep until next update
                print(f"Next update in {interval} seconds...")
                time.sleep(interval)
                iteration += 1
        
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in continuous monitoring: {e}")
        sys.exit(1)


def cleanup_database(optimizer: DBPerformanceOptimizer, days: int, dry_run: bool) -> None:
    """
    Clean up old records from the database.
    
    Args:
        optimizer: DBPerformanceOptimizer instance
        days: Number of days to keep
        dry_run: Whether to just show what would be deleted without actually deleting
    """
    try:
        print(f"\n=== Database Cleanup (Keeping {days} days) ===")
        if dry_run:
            print("DRY RUN: No records will be actually deleted\n")
        
        result = optimizer.cleanup_old_records(older_than_days=days, dry_run=dry_run)
        
        if result:
            total_count = 0
            total_deleted = 0
            
            for table, stats in result.items():
                count = stats.get("count", 0)
                deleted = stats.get("deleted", 0)
                
                total_count += count
                if not dry_run:
                    total_deleted += deleted
                
                if dry_run:
                    print(f"Table {table}: Would delete {count} records")
                else:
                    print(f"Table {table}: Deleted {deleted}/{count} records")
            
            # Show summary
            if dry_run:
                print(f"\nSummary: Would delete {total_count} records across {len(result)} tables")
            else:
                print(f"\nSummary: Deleted {total_deleted} records across {len(result)} tables")
                
                # Run vacuum if records were deleted
                if total_deleted > 0:
                    print("\nRunning vacuum to reclaim space...")
                    optimizer.optimize_database()
                    
                    # Show recovered space
                    before_stats = optimizer.get_database_stats()
                    before_size = before_stats.get("file_size_bytes", 0)
                    after_stats = optimizer.get_database_stats()
                    after_size = after_stats.get("file_size_bytes", 0)
                    
                    if before_size > after_size:
                        saved = before_size - after_size
                        saved_mb = saved / 1024 / 1024
                        print(f"Recovered {saved_mb:.2f} MB of disk space")
        else:
            print("No records found to clean up")
    except Exception as e:
        logger.error(f"Error cleaning up database: {e}")
        sys.exit(1)


def backup_database(optimizer: DBPerformanceOptimizer, backup_path: Optional[str] = None, compress: bool = False) -> None:
    """
    Create a backup of the database.
    
    Args:
        optimizer: DBPerformanceOptimizer instance
        backup_path: Path to store the backup
        compress: Whether to compress the backup
    """
    try:
        print("\n=== Creating Database Backup ===")
        
        # Generate backup path if not provided
        if not backup_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(os.path.dirname(optimizer.db_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"{os.path.basename(optimizer.db_path)}.{timestamp}.bak")
        
        # Create backup
        result = optimizer.backup_database(backup_path=backup_path)
        
        if result:
            print(f"\n✅ Database backup created successfully")
            print(f"Backup location: {result}")
            
            # Compress backup if requested
            if compress and result:
                try:
                    import zipfile
                    
                    # Create zip archive
                    zip_path = f"{result}.zip"
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(result, os.path.basename(result))
                    
                    # Remove original backup
                    os.unlink(result)
                    
                    print(f"Backup compressed to: {zip_path}")
                    
                    # Get compression stats
                    original_size = os.path.getsize(result) if os.path.exists(result) else 0
                    compressed_size = os.path.getsize(zip_path)
                    ratio = original_size / compressed_size if compressed_size > 0 else 0
                    
                    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
                    print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
                    print(f"Compression ratio: {ratio:.2f}x")
                
                except Exception as compress_error:
                    print(f"Warning: Could not compress backup: {compress_error}")
        else:
            print(f"\n❌ Failed to create database backup")
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        sys.exit(1)


def restore_database(optimizer: DBPerformanceOptimizer, backup_path: str) -> None:
    """
    Restore a database from backup.
    
    Args:
        optimizer: DBPerformanceOptimizer instance
        backup_path: Path to the backup file
    """
    try:
        print(f"\n=== Restoring Database from Backup ===")
        print(f"Backup path: {backup_path}")
        print(f"Target database: {optimizer.db_path}")
        
        # Check if backup file exists
        if not os.path.exists(backup_path):
            print(f"\n❌ Backup file not found: {backup_path}")
            return
        
        # Confirm restoration
        confirm = input("\nThis will overwrite the current database. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Restoration cancelled")
            return
        
        # Restore from backup
        result = optimizer.restore_database(backup_path)
        
        if result:
            print(f"\n✅ Database restored successfully from backup")
            print(f"Original database backed up at: {optimizer.db_path}.prerestorebak")
            
            # Show database statistics
            db_stats = optimizer.get_database_stats()
            print(f"\nRestored database statistics:")
            print(f"File size: {db_stats.get('file_size_mb', 0):.2f} MB")
            print(f"Total records: {db_stats.get('total_records', 0)}")
            print(f"Tables: {len(db_stats.get('tables', {}))}")
            print(f"Indexes: {db_stats.get('index_count', 0)}")
        else:
            print(f"\n❌ Failed to restore database from backup")
    except Exception as e:
        logger.error(f"Error restoring database: {e}")
        sys.exit(1)


def db_stats(optimizer: DBPerformanceOptimizer, format_output: str = "text", output_file: Optional[str] = None) -> None:
    """
    Show database statistics.
    
    Args:
        optimizer: DBPerformanceOptimizer instance
        format_output: Output format (text, json, markdown, html)
        output_file: Optional file path to write output to
    """
    try:
        # Get database statistics
        stats = optimizer.get_database_stats()
        
        # Prepare output
        if format_output.lower() == "json":
            # JSON output
            formatted_output = json.dumps(stats, indent=2)
        
        elif format_output.lower() == "markdown":
            # Markdown output
            formatted_output = f"# Database Statistics\n\n"
            formatted_output += f"**Database Path:** {stats.get('database_path', 'unknown')}\n"
            formatted_output += f"**File Size:** {stats.get('file_size_mb', 0):.2f} MB\n"
            formatted_output += f"**Total Records:** {stats.get('total_records', 0)}\n"
            formatted_output += f"**Index Count:** {stats.get('index_count', 0)}\n\n"
            
            # Table records
            formatted_output += "## Table Record Counts\n\n"
            formatted_output += "| Table | Records |\n"
            formatted_output += "|-------|--------|\n"
            
            for table, count in stats.get("tables", {}).items():
                formatted_output += f"| {table} | {count} |\n"
            
            # Cache statistics if available
            if "cache_stats" in stats and stats.get("caching_enabled", False):
                cache_stats = stats.get("cache_stats", {})
                formatted_output += "\n## Cache Statistics\n\n"
                formatted_output += f"**Size:** {cache_stats.get('size', 0)} / {cache_stats.get('max_size', 0)}\n"
                formatted_output += f"**Hits:** {cache_stats.get('hits', 0)}\n"
                formatted_output += f"**Misses:** {cache_stats.get('misses', 0)}\n"
                formatted_output += f"**Hit Ratio:** {cache_stats.get('hit_ratio', 0):.2%}\n"
            
            # Indexes
            if "indexes" in stats:
                formatted_output += "\n## Indexes\n\n"
                for idx in stats.get("indexes", []):
                    formatted_output += f"- {idx}\n"

        elif format_output.lower() == "html":
            # HTML output
            formatted_output = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Database Statistics</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .stats-table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .stats-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .stats-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #4CAF50; color: white; }}
                    .stats-section {{ margin-top: 20px; }}
                    .stats-item {{ margin: 5px 0; }}
                </style>
            </head>
            <body>
                <h1>Database Statistics</h1>
                
                <div class="stats-section">
                    <p class="stats-item"><strong>Database Path:</strong> {stats.get('database_path', 'unknown')}</p>
                    <p class="stats-item"><strong>File Size:</strong> {stats.get('file_size_mb', 0):.2f} MB</p>
                    <p class="stats-item"><strong>Total Records:</strong> {stats.get('total_records', 0)}</p>
                    <p class="stats-item"><strong>Index Count:</strong> {stats.get('index_count', 0)}</p>
                </div>
                
                <h2>Table Record Counts</h2>
                <table class="stats-table">
                    <tr>
                        <th>Table</th>
                        <th>Records</th>
                    </tr>
            """
            
            # Add table rows
            for table, count in stats.get("tables", {}).items():
                formatted_output += f"""
                    <tr>
                        <td>{table}</td>
                        <td>{count}</td>
                    </tr>
                """
            
            # Close the table
            formatted_output += """
                </table>
            """
            
            # Cache statistics if available
            if "cache_stats" in stats and stats.get("caching_enabled", False):
                cache_stats = stats.get("cache_stats", {})
                formatted_output += f"""
                <h2>Cache Statistics</h2>
                <div class="stats-section">
                    <p class="stats-item"><strong>Size:</strong> {cache_stats.get('size', 0)} / {cache_stats.get('max_size', 0)}</p>
                    <p class="stats-item"><strong>Hits:</strong> {cache_stats.get('hits', 0)}</p>
                    <p class="stats-item"><strong>Misses:</strong> {cache_stats.get('misses', 0)}</p>
                    <p class="stats-item"><strong>Hit Ratio:</strong> {cache_stats.get('hit_ratio', 0):.2%}</p>
                </div>
                """
            
            # Indexes
            if "indexes" in stats:
                formatted_output += """
                <h2>Indexes</h2>
                <ul>
                """
                
                for idx in stats.get("indexes", []):
                    formatted_output += f"<li>{idx}</li>\n"
                
                formatted_output += """
                </ul>
                """
            
            # Close HTML
            formatted_output += """
            </body>
            </html>
            """
        
        else:
            # Default text output
            formatted_output = "\nDatabase Statistics:\n"
            formatted_output += f"- Path: {stats.get('database_path', 'unknown')}\n"
            formatted_output += f"- Size: {stats.get('file_size_mb', 0):.2f} MB\n"
            formatted_output += f"- Total Records: {stats.get('total_records', 0)}\n"
            formatted_output += f"- Number of Indexes: {stats.get('index_count', 0)}\n"
            
            if "tables" in stats:
                formatted_output += "\nTable Record Counts:\n"
                for table, count in stats.get("tables", {}).items():
                    formatted_output += f"- {table}: {count}\n"
            
            if "cache_stats" in stats:
                cache_stats = stats.get("cache_stats", {})
                formatted_output += "\nCache Statistics:\n"
                formatted_output += f"- Enabled: {stats.get('caching_enabled', False)}\n"
                formatted_output += f"- Size: {cache_stats.get('size', 0)} / {cache_stats.get('max_size', 0)}\n"
                formatted_output += f"- Hits: {cache_stats.get('hits', 0)}\n"
                formatted_output += f"- Misses: {cache_stats.get('misses', 0)}\n"
                formatted_output += f"- Hit Ratio: {cache_stats.get('hit_ratio', 0):.2%}\n"
            
            if "indexes" in stats:
                formatted_output += "\nIndexes:\n"
                for idx in stats.get("indexes", []):
                    formatted_output += f"- {idx}\n"
        
        # Output to file or print to console
        if output_file:
            with open(output_file, 'w') as f:
                f.write(formatted_output)
            print(f"Database statistics saved to {output_file}")
        else:
            print(formatted_output)
            
    except Exception as e:
        logger.error(f"Error showing database statistics: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Database Performance Monitoring Tool for Simulation Accuracy and Validation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable query caching")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--output", type=str,
                       help="Output file path for applicable commands")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Show metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Show performance metrics")
    metrics_parser.add_argument("--format", choices=["text", "json", "markdown", "html"], default="text",
                              help="Output format")
    metrics_parser.add_argument("--output", type=str,
                              help="Output file path")
    
    # Show database statistics command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--format", choices=["text", "json", "markdown", "html"], default="text",
                            help="Output format")
    stats_parser.add_argument("--output", type=str,
                            help="Output file path")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize database performance")
    optimize_parser.add_argument("--compare", action="store_true",
                               help="Show before and after comparison")
    
    # Dashboard commands
    dashboard_parser = subparsers.add_parser("dashboard", help="Create performance dashboard")
    dashboard_parser.add_argument("--url", required=True, help="Dashboard API URL")
    dashboard_parser.add_argument("--api-key", required=True, help="Dashboard API key")
    dashboard_parser.add_argument("--title", default="Database Performance Dashboard",
                                help="Dashboard title")
    dashboard_parser.add_argument("--style", choices=["detailed", "compact", "overview"],
                                default="detailed", help="Visualization style")
    dashboard_parser.add_argument("--metrics", nargs="+",
                                help="Specific metrics to include (default: all)")
    dashboard_parser.add_argument("--update-interval", type=int, default=3600,
                                help="Update interval in seconds for auto-update")
    dashboard_parser.add_argument("--no-alerts", action="store_true",
                                help="Disable alert creation")
    dashboard_parser.add_argument("--no-auto-update", action="store_true",
                                help="Disable automatic updates")
    
    # Update dashboard command
    update_parser = subparsers.add_parser("update", help="Update dashboard with latest metrics")
    update_parser.add_argument("--url", required=True, help="Dashboard API URL")
    update_parser.add_argument("--api-key", required=True, help="Dashboard API key")
    update_parser.add_argument("--dashboard-id", required=True, help="Dashboard ID to update")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Continuously monitor and update dashboard")
    monitor_parser.add_argument("--url", required=True, help="Dashboard API URL")
    monitor_parser.add_argument("--api-key", required=True, help="Dashboard API key")
    monitor_parser.add_argument("--dashboard-id", required=True, help="Dashboard ID to update")
    monitor_parser.add_argument("--interval", type=int, default=300,
                              help="Update interval in seconds")
    monitor_parser.add_argument("--duration", type=int,
                              help="Duration in seconds (default: indefinite)")
    monitor_parser.add_argument("--quiet", action="store_true",
                              help="Don't print metrics to console")
    
    # Complete solution command
    solution_parser = subparsers.add_parser("solution",
                                          help="Create complete monitoring solution")
    solution_parser.add_argument("--url", required=True, help="Dashboard API URL")
    solution_parser.add_argument("--api-key", required=True, help="Dashboard API key")
    solution_parser.add_argument("--title", default="Complete Monitoring Solution",
                               help="Dashboard title")
    solution_parser.add_argument("--hardware", nargs="+", default=["cpu", "cuda"],
                               help="Hardware types to monitor")
    solution_parser.add_argument("--models", nargs="+", default=["bert", "vit"],
                               help="Model types to monitor")
    solution_parser.add_argument("--performance-metrics", nargs="+",
                               help="Specific performance metrics to include (default: standard metrics)")
    solution_parser.add_argument("--database-metrics", nargs="+",
                               help="Specific database metrics to include (default: all)")
    solution_parser.add_argument("--style", choices=["detailed", "compact", "overview"],
                               default="detailed", help="Visualization style")
    solution_parser.add_argument("--refresh-interval", type=int, default=300,
                               help="Refresh interval in seconds")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old records")
    cleanup_parser.add_argument("--days", type=int, default=90,
                              help="Number of days to keep (records older than this will be deleted)")
    cleanup_parser.add_argument("--dry-run", action="store_true",
                              help="Show what would be deleted without actually deleting")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create database backup")
    backup_parser.add_argument("--output", type=str, help="Path to store the backup")
    backup_parser.add_argument("--compress", action="store_true",
                             help="Compress the backup")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore database from backup")
    restore_parser.add_argument("--backup", required=True, type=str,
                              help="Path to the backup file")
    
    # Auto command (Automated Optimization Manager)
    auto_parser = subparsers.add_parser("auto", help="Automated database optimization")
    auto_parser.add_argument("--action", choices=["check", "optimize", "comprehensive", "monitor", "trends", "recommendations"],
                           default="check", help="Automated action to perform")
    auto_parser.add_argument("--auto-apply", action="store_true",
                           help="Automatically apply optimization actions")
    auto_parser.add_argument("--config", type=str,
                           help="Path to configuration file for the automated optimizer")
    auto_parser.add_argument("--check-interval", type=int, default=3600,
                           help="Interval in seconds between automated checks")
    auto_parser.add_argument("--monitor-time", type=int,
                           help="Time in seconds to run monitoring (default: indefinite)")
    auto_parser.add_argument("--metrics", nargs="+",
                           help="Specific metrics to check")
    auto_parser.add_argument("--days", type=int, default=7,
                           help="Number of days for trend analysis")
    auto_parser.add_argument("--log-file", type=str,
                           help="Log file for automated optimizer")
    auto_parser.add_argument("--retention-days", type=int, default=90,
                           help="Days to retain data when cleaning up old records")
    
    # Predictive command (Database Predictive Analytics)
    predictive_parser = subparsers.add_parser("predictive", help="Predictive analytics for database performance")
    predictive_parser.add_argument("--action", choices=["forecast", "visualize", "alerts", "recommend", "analyze"],
                                default="analyze", help="Predictive action to perform")
    predictive_parser.add_argument("--horizon", choices=["short_term", "medium_term", "long_term"],
                                default="medium_term", help="Forecast horizon")
    predictive_parser.add_argument("--metrics", nargs="+",
                                help="Specific metrics to analyze (default: all monitored metrics)")
    predictive_parser.add_argument("--config", type=str,
                                help="Path to configuration file for predictive analytics")
    predictive_parser.add_argument("--format", choices=["text", "json", "markdown", "html"],
                                default="text", help="Output format for results")
    predictive_parser.add_argument("--output", type=str,
                                help="Output file path for results")
    predictive_parser.add_argument("--visualize", action="store_true",
                                help="Generate visualizations")
    predictive_parser.add_argument("--visual-format", choices=["base64", "file", "object"],
                                default="base64", help="Visualization output format")
    predictive_parser.add_argument("--visual-dir", type=str, default="./visualizations",
                                help="Directory to save visualizations (when using file format)")
    predictive_parser.add_argument("--theme", choices=["light", "dark"],
                                default="light", help="Visualization theme")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize optimizer
    optimizer = get_optimizer(args.db_path, not args.no_cache)
    
    # Execute command
    if args.command == "metrics":
        output_file = args.output
        show_metrics(optimizer, args.format, output_file)
    
    elif args.command == "stats":
        output_file = args.output
        db_stats(optimizer, args.format, output_file)
    
    elif args.command == "optimize":
        optimize_database(optimizer, args.compare)
    
    elif args.command == "dashboard":
        connector = get_dashboard_connector(args.url, args.api_key, optimizer)
        create_dashboard(
            connector,
            args.title,
            args.style,
            not args.no_alerts,
            not args.no_auto_update,
            args.metrics,
            args.update_interval
        )
    
    elif args.command == "update":
        connector = get_dashboard_connector(args.url, args.api_key, optimizer)
        update_dashboard(connector, args.dashboard_id)
    
    elif args.command == "monitor":
        connector = get_dashboard_connector(args.url, args.api_key, optimizer)
        continuous_monitoring(
            connector,
            args.dashboard_id,
            args.interval,
            args.duration,
            not args.quiet
        )
    
    elif args.command == "solution":
        connector = get_dashboard_connector(args.url, args.api_key, optimizer)
        create_complete_solution(
            connector,
            args.title,
            args.hardware,
            args.models,
            args.style,
            args.performance_metrics,
            args.database_metrics,
            args.refresh_interval
        )
    
    elif args.command == "cleanup":
        cleanup_database(optimizer, args.days, args.dry_run)
    
    elif args.command == "backup":
        backup_database(optimizer, args.output, args.compress)
    
    elif args.command == "restore":
        restore_database(optimizer, args.backup)
    
    elif args.command == "auto":
        # Create the automated optimization manager
        auto_manager = get_optimization_manager(
            db_optimizer=optimizer,
            config_file=args.config,
            auto_apply=args.auto_apply
        )
        
        # Update configuration if provided
        if args.check_interval:
            auto_manager.check_interval = args.check_interval
        
        if args.log_file:
            # Add file handler for logging
            file_handler = logging.FileHandler(args.log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            auto_manager.log_file = args.log_file
        
        if args.retention_days:
            auto_manager.retention_days = args.retention_days
        
        # Execute the requested action
        if args.action == "check":
            result = auto_manager.check_performance(args.metrics)
            
            # Print summary of results
            print(f"\nDatabase Status: {result['overall_status'].upper()}")
            print(f"Timestamp: {result['timestamp']}")
            
            if result["issues"]:
                print(f"\nDetected {len(result['issues'])} performance issues:")
                for metric_name, issue in result["issues"].items():
                    print(f"- {metric_name}: {issue['value']} {issue['unit']} "
                           f"(Threshold: {issue['threshold']} {issue['unit']}, Severity: {issue['severity']})")
                    if "recommended_actions" in issue:
                        print(f"  Recommended actions: {', '.join(issue['recommended_actions'])}")
            else:
                print("\nNo performance issues detected.")
        
        elif args.action == "optimize":
            result = auto_manager.optimize_now()
            
            # Print summary of results
            print(f"\nOptimization Status: {result['status'].upper()}")
            print(f"Overall status before: {result['before_overall_status'].upper()}")
            print(f"Overall status after: {result['after_overall_status'].upper()}")
            
            if "results" in result and result["results"]:
                print("\nOptimization results:")
                for metric_name, metric_result in result["results"].items():
                    print(f"- {metric_name}:")
                    if "before_value" in metric_result and "after_value" in metric_result:
                        print(f"  Value: {metric_result['before_value']} → {metric_result['after_value']}")
                    if "improvement" in metric_result and metric_result["improvement"] is not None:
                        print(f"  Improvement: {metric_result['improvement']:.2f}%")
                    if "actions_taken" in metric_result:
                        print(f"  Actions: {', '.join(metric_result['actions_taken'])}")
            else:
                print("\nNo optimizations performed.")
        
        elif args.action == "comprehensive":
            result = auto_manager.run_comprehensive_optimization()
            
            # Print summary of results
            print(f"\nComprehensive Optimization Status: {result['status'].upper()}")
            print(f"Overall status before: {result['before_status'].upper()}")
            print(f"Overall status after: {result['after_status'].upper()}")
            
            if "improvements" in result and result["improvements"]:
                print("\nImprovements:")
                for metric_name, improvement in result["improvements"].items():
                    if improvement["improvement"] is not None:
                        direction = "improved" if improvement["improvement"] > 0 else "worsened"
                        print(f"- {metric_name}: {improvement['before']} → {improvement['after']} "
                              f"({abs(improvement['improvement']):.2f}% {direction})")
            
            if "action_results" in result:
                print("\nAction results:")
                for action, action_result in result["action_results"].items():
                    print(f"- {action}: {action_result['status']}")
        
        elif args.action == "trends":
            result = auto_manager.analyze_trends(days=args.days)
            
            # Print summary of results
            print(f"\nTrend Analysis Status: {result['status'].upper()}")
            print(f"Period: {result['period_days']} days")
            
            if "warnings" in result and result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"- {warning}")
            
            if "trends" in result and result["trends"]:
                print("\nTrends:")
                for metric_name, trend in result["trends"].items():
                    direction_symbol = "↑" if trend["direction"] == "increasing" else "↓" if trend["direction"] == "decreasing" else "→"
                    concern = " (CONCERNING)" if trend["concerning"] else ""
                    print(f"- {metric_name}: {direction_symbol} {trend['change_percent']:.2f}%{concern}")
                    print(f"  Start: {trend['start_value']}, End: {trend['end_value']}")
                    print(f"  Min: {trend['min_value']}, Max: {trend['max_value']}, Avg: {trend['avg_value']:.2f}")
        
        elif args.action == "recommendations":
            result = auto_manager.get_optimization_recommendations()
            
            # Print summary of results
            print(f"\nDatabase Status: {result['overall_status'].upper()}")
            print(f"Timestamp: {result['timestamp']}")
            
            if "recommendations" in result and result["recommendations"]:
                print(f"\nOptimization Recommendations:")
                for metric_name, recommendation in result["recommendations"].items():
                    print(f"- {metric_name}: {recommendation['value']} {recommendation['unit']} "
                          f"(Threshold: {recommendation['threshold']} {recommendation['unit']}, Severity: {recommendation['severity']})")
                    print(f"  Recommended actions: {', '.join(recommendation['recommended_actions'])}")
            else:
                print("\nNo optimization recommendations.")
        
        elif args.action == "monitor":
            # Start continuous monitoring
            monitor_time = args.monitor_time
            
            print(f"\nStarting automated monitoring (interval: {auto_manager.check_interval} seconds)")
            if monitor_time:
                print(f"Will run for {monitor_time} seconds")
            else:
                print("Will run until interrupted (Ctrl+C)")
            
            # Start monitoring
            auto_manager.start_monitoring()
            
            try:
                if monitor_time:
                    time.sleep(monitor_time)
                else:
                    # Run indefinitely until interrupted
                    while True:
                        time.sleep(60)
            except KeyboardInterrupt:
                print("\nMonitoring interrupted by user")
            finally:
                # Stop monitoring
                auto_manager.stop_monitoring()
                
                # Get final results
                result = auto_manager.check_performance()
                
                # Print summary
                print(f"\nFinal Database Status: {result['overall_status'].upper()}")
                
                if result["issues"]:
                    print(f"\nDetected {len(result['issues'])} performance issues:")
                    for metric_name, issue in result["issues"].items():
                        print(f"- {metric_name}: {issue['value']} {issue['unit']} "
                              f"(Threshold: {issue['threshold']} {issue['unit']}, Severity: {issue['severity']})")
                else:
                    print("\nNo performance issues detected.")
        
        # Save results to output file if requested
        if args.output and args.action != "monitor":
            try:
                result_to_save = locals().get('result')
                if result_to_save:
                    with open(args.output, 'w') as f:
                        json.dump(result_to_save, f, indent=2)
                    print(f"\nResults saved to {args.output}")
            except Exception as e:
                print(f"\nError saving results to file: {e}")
    
    elif args.command == "predictive":
        # Create the automated optimization manager first (required for predictive analytics)
        auto_manager = get_optimization_manager(
            db_optimizer=optimizer,
            config_file=args.config
        )
        
        # Create the predictive analytics instance
        predictive = get_predictive_analytics(
            auto_manager=auto_manager,
            config_file=args.config
        )
        
        # If visualize flag is set, update visualization theme
        if args.visualize and args.theme:
            predictive.config["visualization"]["theme"] = args.theme
        
        # Define visual format and directory
        visual_format = args.visual_format
        visual_dir = args.visual_dir
        
        # Run predictive analytics
        run_predictive_analytics(
            predictive_analytics=predictive,
            action=args.action,
            horizon=args.horizon,
            metrics=args.metrics,
            format_output=args.format,
            output_file=args.output,
            generate_visualizations=args.visualize,
            visual_format=visual_format,
            visual_dir=visual_dir,
            theme=args.theme
        )


def run_predictive_analytics(
    predictive_analytics: DatabasePredictiveAnalytics,
    action: str,
    horizon: str,
    metrics: Optional[List[str]] = None,
    format_output: str = "text",
    output_file: Optional[str] = None,
    generate_visualizations: bool = False,
    visual_format: str = "base64",
    visual_dir: str = "./visualizations",
    theme: str = "light"
) -> None:
    """
    Run predictive analytics actions.
    
    Args:
        predictive_analytics: DatabasePredictiveAnalytics instance
        action: Predictive action to perform
        horizon: Forecast horizon
        metrics: Specific metrics to analyze
        format_output: Output format for results
        output_file: Output file path for results
        generate_visualizations: Whether to generate visualizations
        visual_format: Visualization output format
        visual_dir: Directory to save visualizations
        theme: Visualization theme
    """
    # Update visualization config if specified
    if theme != "light":
        predictive_analytics.config["visualization"]["theme"] = theme
    
    if visual_format == "file" and visual_dir:
        # Create directory if it doesn't exist
        os.makedirs(visual_dir, exist_ok=True)
    
    # Execute requested action
    if action == "forecast":
        # Generate forecasts for metrics
        result = predictive_analytics.forecast_database_metrics(
            horizon=horizon,
            specific_metrics=metrics
        )
        
        # Print summary
        print(f"\nForecast Status: {result['status'].upper()}")
        print(f"Horizon: {result['horizon']} ({result['horizon_days']} days)")
        print(f"Timestamp: {result['timestamp']}")
        
        if result["warnings"]:
            print("\nWarnings:")
            for warning in result["warnings"]:
                print(f"- {warning}")
        
        if "forecasts" in result and result["forecasts"]:
            print(f"\nForecasted {len(result['forecasts'])} metrics:")
            for metric_name, forecast in result["forecasts"].items():
                if "trend_analysis" in forecast:
                    trend = forecast["trend_analysis"]
                    direction = trend.get("direction", "stable")
                    magnitude = trend.get("magnitude", "stable")
                    change = trend.get("percent_change", 0.0)
                    
                    if direction == "stable":
                        print(f"- {metric_name}: Stable")
                    else:
                        print(f"- {metric_name}: {magnitude.title()} {direction} trend ({change:.1f}% change)")
        else:
            print("\nNo metrics forecasted.")
        
        # Generate visualizations if requested
        if generate_visualizations:
            vis_result = predictive_analytics.generate_forecast_visualizations(
                forecast_results=result,
                output_format=visual_format
            )
            
            if vis_result["status"] == "success":
                print(f"\nGenerated {len(vis_result['visualizations'])} visualizations")
                
                if visual_format == "file":
                    for metric, vis in vis_result["visualizations"].items():
                        if "filename" in vis:
                            print(f"- {metric}: {vis['filename']}")
            else:
                print(f"\nError generating visualizations: {vis_result.get('message', 'Unknown error')}")
    
    elif action == "visualize":
        # Generate forecasts
        forecast_result = predictive_analytics.forecast_database_metrics(
            horizon=horizon,
            specific_metrics=metrics
        )
        
        # Generate visualizations
        vis_result = predictive_analytics.generate_forecast_visualizations(
            forecast_results=forecast_result,
            output_format=visual_format
        )
        
        # Print summary
        print(f"\nVisualization Status: {vis_result['status'].upper()}")
        print(f"Timestamp: {vis_result['timestamp']}")
        
        if vis_result["status"] == "success":
            print(f"\nGenerated {len(vis_result['visualizations'])} visualizations")
            
            if visual_format == "file":
                for metric, vis in vis_result["visualizations"].items():
                    if "filename" in vis:
                        print(f"- {metric}: {vis['filename']}")
            
            # For base64, we need to save it to a file if an output is provided
            if visual_format == "base64" and output_file:
                try:
                    with open(output_file, 'w') as f:
                        json.dump(vis_result, f, indent=2)
                    print(f"\nVisualization data saved to {output_file}")
                except Exception as e:
                    print(f"\nError saving visualization data to file: {e}")
        else:
            print(f"\nError generating visualizations: {vis_result.get('message', 'Unknown error')}")
    
    elif action == "alerts":
        # Generate forecasts
        forecast_result = predictive_analytics.forecast_database_metrics(
            horizon=horizon,
            specific_metrics=metrics
        )
        
        # Check for alerts
        alert_result = predictive_analytics.check_predicted_thresholds(
            forecast_results=forecast_result
        )
        
        # Print summary
        print(f"\nAlert Status: {alert_result['status'].upper()}")
        print(f"Timestamp: {alert_result['timestamp']}")
        
        if "alerts" in alert_result and alert_result["alerts"]:
            print(f"\nDetected {len(alert_result['alerts'])} potential future alerts:")
            
            # Group alerts by severity
            error_alerts = []
            warning_alerts = []
            
            for metric_name, alerts in alert_result["alerts"].items():
                for alert in alerts:
                    if alert["severity"] == "error":
                        error_alerts.append((metric_name, alert))
                    else:
                        warning_alerts.append((metric_name, alert))
            
            # Print error alerts first
            if error_alerts:
                print("\nError alerts:")
                for metric_name, alert in error_alerts:
                    print(f"- {metric_name}: {alert['message']}")
                    print(f"  Days until: {alert['days_until']}")
            
            # Print warning alerts
            if warning_alerts:
                print("\nWarning alerts:")
                for metric_name, alert in warning_alerts:
                    print(f"- {metric_name}: {alert['message']}")
                    print(f"  Days until: {alert['days_until']}")
        else:
            print("\nNo potential alerts detected.")
    
    elif action == "recommend":
        # Generate forecasts
        forecast_result = predictive_analytics.forecast_database_metrics(
            horizon=horizon,
            specific_metrics=metrics
        )
        
        # Check for alerts
        alert_result = predictive_analytics.check_predicted_thresholds(
            forecast_results=forecast_result
        )
        
        # Generate recommendations
        rec_result = predictive_analytics.recommend_proactive_actions(
            forecast_results=forecast_result,
            threshold_alerts=alert_result
        )
        
        # Print summary
        print(f"\nRecommendation Status: {rec_result['status'].upper()}")
        print(f"Timestamp: {rec_result['timestamp']}")
        
        if "summary" in rec_result and rec_result["summary"]:
            print("\nSummary:")
            for summary_item in rec_result["summary"]:
                print(f"- {summary_item}")
        
        if "recommendations" in rec_result and rec_result["recommendations"]:
            print(f"\nRecommended {len(rec_result['recommendations'])} proactive actions:")
            
            # Sort recommendations by urgency
            urgency_order = {
                "immediate": 0,
                "this_week": 1,
                "this_month": 2,
                "future": 3
            }
            
            sorted_recs = sorted(rec_result["recommendations"], 
                                key=lambda x: urgency_order.get(x["urgency"], 99))
            
            for rec in sorted_recs:
                urgency_display = rec["urgency"].replace("_", " ").title()
                print(f"\n- {rec['metric']} ({urgency_display}, {rec['days_until']} days):")
                print(f"  {rec['message']}")
                if rec["recommended_actions"]:
                    print("  Recommended actions:")
                    for action in rec["recommended_actions"]:
                        print(f"  - {action}")
        else:
            print("\nNo recommendations generated.")
    
    elif action == "analyze":
        # Run comprehensive analysis
        result = predictive_analytics.analyze_database_health_forecast(
            horizon=horizon,
            specific_metrics=metrics,
            generate_visualizations=generate_visualizations,
            output_format=visual_format
        )
        
        # Print summary
        print(f"\nAnalysis Status: {result['status'].upper()}")
        print(f"Timestamp: {result['timestamp']}")
        
        # Print summary stats
        if "summary" in result:
            summary = result["summary"]
            print(f"\nAnalyzed {summary.get('total_metrics_analyzed', 0)} metrics over "
                 f"{summary.get('forecast_horizon_days', 0)} days")
            print(f"Metrics with alerts: {summary.get('metrics_with_alerts', 0)}")
            print(f"Total recommendations: {summary.get('total_recommendations', 0)}")
            
            # Print forecast trends
            if "forecast_trends" in summary and summary["forecast_trends"]:
                print("\nForecast Trends:")
                for trend in summary["forecast_trends"]:
                    print(f"- {trend}")
            
            # Print alert summary
            if "alert_summary" in summary and summary["alert_summary"]:
                print("\nAlert Summary:")
                for alert in summary["alert_summary"]:
                    print(f"- {alert}")
        
        # Print threshold alerts
        if "threshold_alerts" in result and "alerts" in result["threshold_alerts"]:
            alerts = result["threshold_alerts"]["alerts"]
            if alerts:
                print(f"\nDetected {len(alerts)} potential future alerts")
                
                # Count error and warning alerts
                error_count = 0
                warning_count = 0
                
                for metric_name, metric_alerts in alerts.items():
                    for alert in metric_alerts:
                        if alert["severity"] == "error":
                            error_count += 1
                        else:
                            warning_count += 1
                
                print(f"- Error alerts: {error_count}")
                print(f"- Warning alerts: {warning_count}")
        
        # Print recommendations count
        if "recommendations" in result and "recommendations" in result["recommendations"]:
            recs = result["recommendations"]["recommendations"]
            if recs:
                print(f"\nGenerated {len(recs)} proactive recommendations")
                
                # Group by urgency
                urgency_counts = defaultdict(int)
                for rec in recs:
                    urgency_counts[rec["urgency"]] += 1
                
                # Print counts by urgency
                for urgency, count in sorted(urgency_counts.items()):
                    urgency_display = urgency.replace("_", " ").title()
                    print(f"- {urgency_display}: {count}")
        
        # Print visualization info
        if generate_visualizations and "visualizations" in result:
            vis_result = result["visualizations"]
            if vis_result["status"] == "success":
                print(f"\nGenerated {len(vis_result.get('visualizations', {}))} visualizations")
                
                if visual_format == "file":
                    for metric, vis in vis_result.get("visualizations", {}).items():
                        if "filename" in vis:
                            print(f"- {metric}: {vis['filename']}")
            else:
                print(f"\nError generating visualizations: {vis_result.get('message', 'Unknown error')}")
    
    # Save results to output file if requested
    if output_file and format_output.lower() != "text":
        try:
            # Prepare output based on format
            if format_output.lower() == "json":
                output_data = json.dumps(locals().get('result', {}), indent=2)
            elif format_output.lower() == "markdown":
                # Simple markdown format (could be enhanced)
                output_data = f"# Database Predictive Analytics Results\n\n"
                output_data += f"**Action:** {action}\n\n"
                output_data += f"**Status:** {locals().get('result', {}).get('status', 'unknown').upper()}\n\n"
                output_data += f"**Timestamp:** {locals().get('result', {}).get('timestamp', datetime.datetime.now().isoformat())}\n\n"
                
                # Add more markdown based on the action type
                output_data += "## Details\n\n"
                output_data += f"```json\n{json.dumps(locals().get('result', {}), indent=2)}\n```\n"
            elif format_output.lower() == "html":
                # Simple HTML format (could be enhanced)
                output_data = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Database Predictive Analytics Results</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #333; }}
                        .status-success {{ color: green; }}
                        .status-warning {{ color: orange; }}
                        .status-error {{ color: red; }}
                        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    </style>
                </head>
                <body>
                    <h1>Database Predictive Analytics Results</h1>
                    <p><strong>Action:</strong> {action}</p>
                    <p><strong>Status:</strong> <span class="status-{locals().get('result', {}).get('status', 'unknown')}">{locals().get('result', {}).get('status', 'unknown').upper()}</span></p>
                    <p><strong>Timestamp:</strong> {locals().get('result', {}).get('timestamp', datetime.datetime.now().isoformat())}</p>
                    
                    <h2>Details</h2>
                    <pre>{json.dumps(locals().get('result', {}), indent=2)}</pre>
                </body>
                </html>
                """
            else:
                output_data = str(locals().get('result', {}))
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(output_data)
            
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"\nError saving results to file: {e}")


if __name__ == "__main__":
    main()