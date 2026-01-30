#!/usr/bin/env python3
"""
Customizable Dashboard System for the Advanced Visualization System.

This script provides a simplified interface for creating and managing
customizable dashboards with the Advanced Visualization System.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard_system")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Try to import the CustomizableDashboard
try:
    from data.duckdb.visualization.advanced_visualization import CustomizableDashboard
    HAS_DASHBOARD = True
except ImportError as e:
    logger.error(f"Error importing CustomizableDashboard: {e}")
    logger.error("Make sure plotly, pandas, and scikit-learn are installed.")
    logger.error("Install with: pip install plotly pandas scikit-learn")
    HAS_DASHBOARD = False

def create_dashboard(args):
    """Create a new dashboard."""
    if not HAS_DASHBOARD:
        logger.error("CustomizableDashboard is not available.")
        return 1
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(
        db_connection=None, 
        theme=args.theme, 
        debug=args.debug,
        output_dir=args.output_dir
    )
    
    # Create dashboard from template or custom configuration
    try:
        if args.template:
            # Create a dashboard from template
            dashboard_path = dashboard.create_dashboard(
                dashboard_name=args.name,
                template=args.template,
                title=args.title,
                description=args.description,
                columns=args.columns,
                row_height=args.row_height
            )
        else:
            # Create custom dashboard with all visualization types
            components = []
            
            # Add 3D visualization if requested
            if args.include_3d:
                components.append({
                    "type": "3d",
                    "config": {
                        "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                        "dimensions": ["model_family", "hardware_type"],
                        "title": "3D Performance Visualization"
                    },
                    "width": 2,
                    "height": 1
                })
            
            # Add heatmap if requested
            if args.include_heatmap:
                components.append({
                    "type": "heatmap",
                    "config": {
                        "metric": "throughput_items_per_second",
                        "title": "Hardware Throughput Comparison"
                    },
                    "width": 1,
                    "height": 1
                })
                
                components.append({
                    "type": "heatmap",
                    "config": {
                        "metric": "average_latency_ms",
                        "title": "Hardware Latency Comparison"
                    },
                    "width": 1,
                    "height": 1
                })
            
            # Add time-series if requested
            if args.include_timeseries:
                components.append({
                    "type": "time-series",
                    "config": {
                        "metric": "throughput_items_per_second",
                        "dimensions": ["model_family", "hardware_type"],
                        "time_range": 90,
                        "title": "Performance Trends"
                    },
                    "width": 1,
                    "height": 1
                })
            
            # Add animated time-series if requested
            if args.include_animated:
                components.append({
                    "type": "animated-time-series",
                    "config": {
                        "metric": "throughput_items_per_second",
                        "dimensions": ["model_family", "hardware_type"],
                        "time_range": 90,
                        "title": "Animated Performance Trends"
                    },
                    "width": 2,
                    "height": 1
                })
            
            # Create the dashboard
            dashboard_path = dashboard.create_dashboard(
                dashboard_name=args.name,
                title=args.title or "Custom Performance Dashboard",
                description=args.description or "Custom dashboard with visualization components",
                components=components,
                columns=args.columns,
                row_height=args.row_height
            )
        
        if dashboard_path:
            logger.info(f"Dashboard created successfully: {dashboard_path}")
            
            # Open in browser if requested
            if args.open:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                except Exception as e:
                    logger.error(f"Error opening dashboard in browser: {e}")
            
            return 0
        else:
            logger.error("Failed to create dashboard")
            return 1
            
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        return 1

def list_dashboards(args):
    """List all saved dashboards."""
    if not HAS_DASHBOARD:
        logger.error("CustomizableDashboard is not available.")
        return 1
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(output_dir=args.output_dir)
    
    # List all dashboards
    try:
        dashboards = dashboard.list_dashboards()
        
        if not dashboards:
            print("No dashboards found.")
            return 0
        
        # Print dashboard list
        print("\nAvailable dashboards:")
        print("-" * 80)
        print(f"{'Name':<30} {'Title':<30} {'Components':<10} {'Created':<20}")
        print("-" * 80)
        
        for name, details in dashboards.items():
            created = details.get("created_at", "").split("T")[0] if details.get("created_at") else ""
            print(f"{name:<30} {details.get('title', ''):<30} {details.get('components', 0):<10} {created:<20}")
        
        print("-" * 80)
        return 0
    
    except Exception as e:
        logger.error(f"Error listing dashboards: {e}")
        return 1

def list_templates(args):
    """List all available dashboard templates."""
    if not HAS_DASHBOARD:
        logger.error("CustomizableDashboard is not available.")
        return 1
    
    # Create dashboard instance
    dashboard = CustomizableDashboard()
    
    try:
        # List available templates
        templates = dashboard.list_available_templates()
        
        if not templates:
            print("No templates available.")
            return 0
        
        # Print template list
        print("\nAvailable templates:")
        print("-" * 80)
        print(f"{'Name':<20} {'Title':<40} {'Components':<10}")
        print("-" * 80)
        
        for name, details in templates.items():
            print(f"{name:<20} {details.get('title', ''):<40} {details.get('components', 0):<10}")
        
        print("-" * 80)
        
        # List available components
        components = dashboard.list_available_components()
        
        if components:
            print("\nAvailable component types:")
            print("-" * 80)
            print(f"{'Type':<20} {'Description':<60}")
            print("-" * 80)
            
            for comp_type, description in components.items():
                print(f"{comp_type:<20} {description:<60}")
            
            print("-" * 80)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        return 1

def add_component(args):
    """Add a component to an existing dashboard."""
    if not HAS_DASHBOARD:
        logger.error("CustomizableDashboard is not available.")
        return 1
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(output_dir=args.output_dir)
    
    # Load component config from file or use empty dict
    config = {}
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return 1
    
    try:
        # Add component to dashboard
        dashboard_path = dashboard.add_component_to_dashboard(
            dashboard_name=args.name,
            component_type=args.type,
            component_config=config,
            width=args.width,
            height=args.height
        )
        
        if dashboard_path:
            logger.info(f"Component added successfully to dashboard: {dashboard_path}")
            
            # Open in browser if requested
            if args.open:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                except Exception as e:
                    logger.error(f"Error opening dashboard in browser: {e}")
            
            return 0
        else:
            logger.error("Failed to add component to dashboard")
            return 1
    
    except Exception as e:
        logger.error(f"Error adding component: {e}")
        return 1

def update_dashboard(args):
    """Update an existing dashboard."""
    if not HAS_DASHBOARD:
        logger.error("CustomizableDashboard is not available.")
        return 1
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(output_dir=args.output_dir)
    
    try:
        # Update dashboard
        dashboard_path = dashboard.update_dashboard(
            dashboard_name=args.name,
            title=args.title,
            description=args.description,
            columns=args.columns,
            row_height=args.row_height
        )
        
        if dashboard_path:
            logger.info(f"Dashboard updated successfully: {dashboard_path}")
            
            # Open in browser if requested
            if args.open:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                except Exception as e:
                    logger.error(f"Error opening dashboard in browser: {e}")
            
            return 0
        else:
            logger.error("Failed to update dashboard")
            return 1
    
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        return 1

def export_dashboard(args):
    """Export a dashboard to different formats."""
    if not HAS_DASHBOARD:
        logger.error("CustomizableDashboard is not available.")
        return 1
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(output_dir=args.output_dir)
    
    try:
        # Export dashboard
        export_path = dashboard.export_dashboard(
            dashboard_name=args.name,
            format=args.format,
            output_path=args.output_path
        )
        
        if export_path:
            logger.info(f"Dashboard exported successfully: {export_path}")
            
            # Open in browser if requested and format is HTML
            if args.open and args.format == "html":
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(export_path)}")
                except Exception as e:
                    logger.error(f"Error opening exported dashboard in browser: {e}")
            
            return 0
        else:
            logger.error("Failed to export dashboard")
            return 1
    
    except Exception as e:
        logger.error(f"Error exporting dashboard: {e}")
        return 1

def delete_dashboard(args):
    """Delete a dashboard."""
    if not HAS_DASHBOARD:
        logger.error("CustomizableDashboard is not available.")
        return 1
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(output_dir=args.output_dir)
    
    # Confirm deletion if not forced
    if not args.force:
        confirm = input(f"Are you sure you want to delete dashboard '{args.name}'? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Dashboard deletion cancelled.")
            return 0
    
    try:
        # Delete dashboard
        success = dashboard.delete_dashboard(args.name)
        
        if success:
            logger.info(f"Dashboard '{args.name}' deleted successfully.")
            return 0
        else:
            logger.error(f"Failed to delete dashboard '{args.name}'.")
            return 1
    
    except Exception as e:
        logger.error(f"Error deleting dashboard: {e}")
        return 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Customizable Dashboard System for the Advanced Visualization System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("--output-dir", default="./dashboards",
                       help="Directory to save dashboards")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create dashboard command
    create_parser = subparsers.add_parser("create", help="Create a new dashboard")
    create_parser.add_argument("--name", type=str, required=True,
                             help="Dashboard name")
    create_parser.add_argument("--template", type=str, choices=["overview", "hardware_comparison", "model_analysis", "empty"],
                             help="Dashboard template to use")
    create_parser.add_argument("--title", type=str,
                             help="Dashboard title")
    create_parser.add_argument("--description", type=str,
                             help="Dashboard description")
    create_parser.add_argument("--columns", type=int, default=2,
                             help="Number of columns in the grid layout")
    create_parser.add_argument("--row-height", type=int, default=500,
                             help="Height of each row in pixels")
    create_parser.add_argument("--theme", type=str, choices=["light", "dark"], default="light",
                             help="Dashboard theme")
    create_parser.add_argument("--include-3d", action="store_true",
                             help="Include 3D visualization component")
    create_parser.add_argument("--include-heatmap", action="store_true",
                             help="Include heatmap visualization components")
    create_parser.add_argument("--include-timeseries", action="store_true",
                             help="Include time-series visualization component")
    create_parser.add_argument("--include-animated", action="store_true",
                             help="Include animated time-series visualization component")
    create_parser.add_argument("--include-all", action="store_true",
                             help="Include all visualization components")
    create_parser.add_argument("--open", action="store_true",
                             help="Open dashboard in browser after creation")
    
    # List dashboards command
    list_parser = subparsers.add_parser("list", help="List all saved dashboards")
    
    # List templates command
    templates_parser = subparsers.add_parser("templates", help="List all available templates and components")
    
    # Add component command
    add_parser = subparsers.add_parser("add", help="Add a component to a dashboard")
    add_parser.add_argument("--name", type=str, required=True,
                           help="Dashboard name")
    add_parser.add_argument("--type", type=str, required=True,
                           choices=["3d", "heatmap", "time-series", "animated-time-series"],
                           help="Component type")
    add_parser.add_argument("--config-file", type=str,
                           help="Path to component configuration JSON file")
    add_parser.add_argument("--width", type=int, default=1,
                           help="Component width (number of columns)")
    add_parser.add_argument("--height", type=int, default=1,
                           help="Component height (number of rows)")
    add_parser.add_argument("--open", action="store_true",
                           help="Open dashboard in browser after adding component")
    
    # Update dashboard command
    update_parser = subparsers.add_parser("update", help="Update an existing dashboard")
    update_parser.add_argument("--name", type=str, required=True,
                              help="Dashboard name")
    update_parser.add_argument("--title", type=str,
                              help="Dashboard title")
    update_parser.add_argument("--description", type=str,
                              help="Dashboard description")
    update_parser.add_argument("--columns", type=int,
                              help="Number of columns in the grid layout")
    update_parser.add_argument("--row-height", type=int,
                              help="Height of each row in pixels")
    update_parser.add_argument("--open", action="store_true",
                              help="Open dashboard in browser after update")
    
    # Export dashboard command
    export_parser = subparsers.add_parser("export", help="Export a dashboard to different formats")
    export_parser.add_argument("--name", type=str, required=True,
                              help="Dashboard name")
    export_parser.add_argument("--format", type=str, choices=["html", "png", "pdf"], default="html",
                              help="Export format")
    export_parser.add_argument("--output-path", type=str,
                              help="Output file path")
    export_parser.add_argument("--open", action="store_true",
                              help="Open exported dashboard in browser (HTML only)")
    
    # Delete dashboard command
    delete_parser = subparsers.add_parser("delete", help="Delete a dashboard")
    delete_parser.add_argument("--name", type=str, required=True,
                              help="Dashboard name")
    delete_parser.add_argument("--force", action="store_true",
                              help="Force deletion without confirmation")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process include_all flag
    if hasattr(args, 'include_all') and args.include_all:
        args.include_3d = True
        args.include_heatmap = True
        args.include_timeseries = True
        args.include_animated = True
    
    # Execute command
    if args.command == "create":
        return create_dashboard(args)
    elif args.command == "list":
        return list_dashboards(args)
    elif args.command == "templates":
        return list_templates(args)
    elif args.command == "add":
        return add_component(args)
    elif args.command == "update":
        return update_dashboard(args)
    elif args.command == "export":
        return export_dashboard(args)
    elif args.command == "delete":
        return delete_dashboard(args)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())