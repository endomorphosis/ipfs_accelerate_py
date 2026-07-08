#!/usr/bin/env python3
"""
Advanced Visualization Export Tool

This script provides a command-line interface for testing and using
the export capabilities of the Advanced Visualization System.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("export_tool")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import database API
from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI

# Import visualization system
try:
    from data.duckdb.visualization.advanced_visualization import AdvancedVisualizationSystem
    HAS_ADVANCED_VISUALIZATION = True
except ImportError as e:
    logger.error(f"Error importing AdvancedVisualizationSystem: {e}")
    logger.error("Make sure plotly, pandas, and scikit-learn are installed.")
    logger.error("Install with: pip install plotly pandas scikit-learn")
    HAS_ADVANCED_VISUALIZATION = False


def export_visualization(args):
    """Export a specific visualization."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization system not available.")
        return 1
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=args.db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=args.output_dir)
    viz.configure({"auto_open": not args.no_open, "theme": args.theme})
    
    # Create and export the visualization based on the type
    try:
        if args.viz_type == '3d':
            # Create 3D visualization
            result = viz.create_3d_performance_visualization(
                metrics=args.metrics or ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                dimensions=args.dimensions or ["model_family", "hardware_type"],
                title=args.title or "3D Performance Visualization"
            )
            
            # Export the visualization
            exports = viz.export_3d_visualization(
                visualization_data=result,
                formats=args.formats.split(',') if args.formats else None,
                visualization_id=args.name or "3d_visualization",
                settings={"width": args.width, "height": args.height, "scale": args.scale}
            )
            
            logger.info(f"Exported 3D visualization: {exports}")
            
        elif args.viz_type == 'heatmap':
            # Create hardware comparison heatmap
            result = viz.create_hardware_comparison_heatmap(
                metric=args.metrics[0] if args.metrics else "throughput",
                model_families=args.dimensions[0].split(',') if args.dimensions and len(args.dimensions) > 0 else None,
                hardware_types=args.dimensions[1].split(',') if args.dimensions and len(args.dimensions) > 1 else None,
                batch_size=args.batch_size,
                title=args.title or "Hardware Comparison Heatmap"
            )
            
            # Export the visualization
            exports = viz.export_heatmap_visualization(
                visualization_data=result,
                formats=args.formats.split(',') if args.formats else None,
                visualization_id=args.name or "heatmap_visualization",
                settings={"width": args.width, "height": args.height, "scale": args.scale}
            )
            
            logger.info(f"Exported heatmap visualization: {exports}")
            
        elif args.viz_type == 'power':
            # Create power efficiency visualization
            result = viz.create_power_efficiency_visualization(
                hardware_types=args.dimensions[0].split(',') if args.dimensions and len(args.dimensions) > 0 else None,
                model_families=args.dimensions[1].split(',') if args.dimensions and len(args.dimensions) > 1 else None,
                batch_sizes=[int(b) for b in args.dimensions[2].split(',')] if args.dimensions and len(args.dimensions) > 2 else None,
                title=args.title or "Power Efficiency Visualization"
            )
            
            # Export the visualization
            exports = viz.export_power_visualization(
                visualization_data=result,
                formats=args.formats.split(',') if args.formats else None,
                visualization_id=args.name or "power_visualization",
                settings={"width": args.width, "height": args.height, "scale": args.scale}
            )
            
            logger.info(f"Exported power visualization: {exports}")
            
        elif args.viz_type == 'time-series':
            # Create animated time series visualization
            result = viz.create_animated_time_series_visualization(
                metric=args.metrics[0] if args.metrics else "throughput_items_per_second",
                dimensions=args.dimensions or ["model_family", "hardware_type"],
                time_range=args.time_range,
                time_interval=args.time_interval,
                include_trend=not args.no_trend,
                window_size=args.window_size,
                title=args.title or "Performance Trends Over Time"
            )
            
            # Export the visualization
            exports = viz.export_time_series_visualization(
                visualization_data=result,
                formats=args.formats.split(',') if args.formats else None,
                visualization_id=args.name or "time_series_visualization",
                settings={
                    "width": args.width, 
                    "height": args.height, 
                    "scale": args.scale,
                    "fps": args.fps,
                    "duration": args.duration * 1000,  # Convert to milliseconds
                    "frame_duration": args.frame_duration
                },
                include_animation=not args.no_animation
            )
            
            logger.info(f"Exported time-series visualization: {exports}")
            
        elif args.viz_type == 'dashboard':
            # Create or use an existing dashboard
            dashboard_name = args.name or "test_dashboard"
            
            if not viz.get_dashboard(dashboard_name):
                # Create a new dashboard
                viz.create_dashboard(
                    dashboard_name=dashboard_name,
                    template=args.dashboard_template,
                    title=args.title or f"{dashboard_name.replace('_', ' ').title()} Dashboard",
                    description=args.description or f"Dashboard created for export testing.",
                    columns=args.columns,
                    row_height=args.row_height
                )
            
            # Export the dashboard
            output_path = viz.export_dashboard(
                dashboard_name=dashboard_name,
                format=args.format,
                output_path=args.export_path
            )
            
            logger.info(f"Exported dashboard to: {output_path}")
            
        else:
            logger.error(f"Unsupported visualization type: {args.viz_type}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating or exporting visualization: {e}")
        return 1


def export_all(args):
    """Export all visualization types."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization system not available.")
        return 1
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=args.db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=args.output_dir)
    viz.configure({"auto_open": not args.no_open, "theme": args.theme})
    
    try:
        # Create visualizations of each type
        visualizations = {}
        
        # 3D visualization
        result_3d = viz.create_3d_performance_visualization(
            metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
            dimensions=["model_family", "hardware_type"],
            title="3D Performance Visualization"
        )
        visualizations["3d_visualization"] = result_3d
        
        # Heatmap visualization
        result_heatmap = viz.create_hardware_comparison_heatmap(
            metric="throughput",
            title="Hardware Comparison Heatmap"
        )
        visualizations["heatmap_visualization"] = result_heatmap
        
        # Power efficiency visualization
        result_power = viz.create_power_efficiency_visualization(
            title="Power Efficiency Visualization"
        )
        visualizations["power_visualization"] = result_power
        
        # Animated time series visualization
        result_time_series = viz.create_animated_time_series_visualization(
            metric="throughput_items_per_second",
            dimensions=["model_family", "hardware_type"],
            time_range=90,
            time_interval="day",
            include_trend=True,
            window_size=5,
            title="Performance Trends Over Time"
        )
        visualizations["time_series_visualization"] = result_time_series
        
        # Export all visualizations
        exports = viz.export_all_visualizations(
            visualizations=visualizations,
            output_dir=args.output_dir,
            formats={
                '3d': args.formats.split(',') if args.formats else ['html', 'png', 'pdf', 'json'],
                'heatmap': args.formats.split(',') if args.formats else ['html', 'png', 'pdf', 'json'],
                'power': args.formats.split(',') if args.formats else ['html', 'png', 'pdf', 'json'],
                'time-series': args.formats.split(',') if args.formats else ['html', 'png', 'pdf', 'json', 'mp4', 'gif']
            },
            settings={
                "width": args.width,
                "height": args.height,
                "scale": args.scale,
                "fps": args.fps,
                "duration": args.duration * 1000,  # Convert to milliseconds
                "frame_duration": args.frame_duration
            },
            create_index=True,
            title="All Visualization Types"
        )
        
        # Generate an export report
        report_path = viz.generate_export_report(
            title="Visualization Export Report",
            description="Comprehensive report of all exported visualizations"
        )
        
        logger.info(f"Exported all visualizations. Report available at: {report_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error creating or exporting visualizations: {e}")
        return 1


def export_animation(args):
    """Export an animated time-series visualization with optimized settings."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization system not available.")
        return 1
    
    # Initialize database API
    db_api = BenchmarkDBAPI(db_path=args.db_path)
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(db_api=db_api, output_dir=args.output_dir)
    viz.configure({"auto_open": not args.no_open, "theme": args.theme})
    
    try:
        # Create animated time series visualization
        result = viz.create_animated_time_series_visualization(
            metric=args.metric,
            dimensions=args.dimensions,
            time_range=args.time_range,
            time_interval=args.time_interval,
            include_trend=not args.no_trend,
            window_size=args.window_size,
            title=args.title or "Performance Trends Over Time"
        )
        
        # Export the animation
        output_path = viz.export_animated_time_series(
            visualization_data=result,
            format=args.format,
            visualization_id=args.name or "animated_time_series",
            settings={
                "width": args.width,
                "height": args.height,
                "scale": args.scale,
                "fps": args.fps,
                "duration": args.duration * 1000,  # Convert to milliseconds
                "frame_duration": args.frame_duration,
                "transition_duration": args.transition_duration
            }
        )
        
        logger.info(f"Exported animation to: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error creating or exporting animation: {e}")
        return 1


def configure_export(args):
    """Configure export settings."""
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization system not available.")
        return 1
    
    # Initialize visualization system
    viz = AdvancedVisualizationSystem(output_dir=args.output_dir)
    
    # Configure export settings
    settings = {
        "width": args.width,
        "height": args.height,
        "scale": args.scale,
        "include_plotlyjs": not args.no_plotlyjs,
        "include_mathjax": args.include_mathjax,
        "full_html": not args.no_full_html
    }
    
    # Add animation settings if provided
    if args.fps:
        settings["fps"] = args.fps
    if args.duration:
        settings["duration"] = args.duration * 1000  # Convert to milliseconds
    if args.frame_duration:
        settings["frame_duration"] = args.frame_duration
    if args.transition_duration:
        settings["transition_duration"] = args.transition_duration
    
    # Configure export settings
    viz.configure_export_settings(settings)
    
    logger.info(f"Configured export settings: {settings}")
    return 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced Visualization Export Tool")
    
    # Global arguments
    parser.add_argument("--output-dir", default="./exports",
                       help="Directory to save exports")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--no-open", action="store_true",
                       help="Don't automatically open visualizations in browser")
    parser.add_argument("--theme", choices=["light", "dark"], default="light",
                       help="Theme for visualizations")
    
    # Common export arguments
    export_args = argparse.ArgumentParser(add_help=False)
    export_args.add_argument("--name", 
                            help="Name for the exported visualization")
    export_args.add_argument("--title",
                            help="Title for the visualization")
    export_args.add_argument("--formats",
                            help="Comma-separated list of export formats (html,png,pdf,json,mp4,gif)")
    export_args.add_argument("--width", type=int, default=1200,
                            help="Width of the exported visualization")
    export_args.add_argument("--height", type=int, default=800,
                            help="Height of the exported visualization")
    export_args.add_argument("--scale", type=float, default=2.0,
                            help="Scale factor for the exported visualization")
    
    # Animation arguments
    animation_args = argparse.ArgumentParser(add_help=False)
    animation_args.add_argument("--fps", type=int, default=30,
                               help="Frames per second for animations")
    animation_args.add_argument("--duration", type=int, default=10,
                               help="Duration of animations in seconds")
    animation_args.add_argument("--frame-duration", type=int, default=50,
                               help="Duration of each frame in milliseconds")
    animation_args.add_argument("--transition-duration", type=int, default=100,
                               help="Duration of transitions in milliseconds")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Export visualization command
    export_parser = subparsers.add_parser("export", help="Export a specific visualization",
                                        parents=[export_args, animation_args])
    export_parser.add_argument("--viz-type", required=True, 
                              choices=["3d", "heatmap", "power", "time-series", "dashboard"],
                              help="Type of visualization to export")
    export_parser.add_argument("--metrics", nargs="+",
                              help="Performance metrics to visualize")
    export_parser.add_argument("--dimensions", nargs="+",
                              help="Dimensions for grouping (model_family, hardware_type, etc.)")
    export_parser.add_argument("--time-range", type=int, default=90,
                              help="Time range in days (for time-series)")
    export_parser.add_argument("--time-interval", choices=["hour", "day", "week", "month"],
                              default="day", help="Time interval (for time-series)")
    export_parser.add_argument("--window-size", type=int, default=5,
                              help="Window size for trend analysis (for time-series)")
    export_parser.add_argument("--no-trend", action="store_true",
                              help="Don't include trend lines (for time-series)")
    export_parser.add_argument("--no-animation", action="store_true",
                              help="Don't include animation formats (for time-series)")
    export_parser.add_argument("--batch-size", type=int, default=1,
                              help="Batch size for heatmap visualization")
    export_parser.add_argument("--dashboard-template", 
                              choices=["overview", "hardware_comparison", "model_analysis", "empty"],
                              help="Template for dashboard creation")
    export_parser.add_argument("--description",
                              help="Description for the dashboard")
    export_parser.add_argument("--columns", type=int, default=2,
                              help="Number of columns in the dashboard grid")
    export_parser.add_argument("--row-height", type=int, default=500,
                              help="Height of each row in pixels")
    export_parser.add_argument("--format", choices=["html", "pdf", "png"], default="html",
                              help="Export format for dashboard")
    export_parser.add_argument("--export-path",
                              help="Path for the exported file")
    
    # Export all command
    export_all_parser = subparsers.add_parser("export-all", help="Export all visualization types",
                                            parents=[export_args, animation_args])
    
    # Export animation command
    export_animation_parser = subparsers.add_parser("export-animation", 
                                                  help="Export an animated time-series visualization",
                                                  parents=[export_args, animation_args])
    export_animation_parser.add_argument("--metric", default="throughput_items_per_second",
                                       help="Performance metric to visualize")
    export_animation_parser.add_argument("--dimensions", nargs="+", default=["model_family", "hardware_type"],
                                       help="Dimensions for grouping (model_family, hardware_type, etc.)")
    export_animation_parser.add_argument("--time-range", type=int, default=90,
                                       help="Time range in days")
    export_animation_parser.add_argument("--time-interval", choices=["hour", "day", "week", "month"],
                                       default="day", help="Time interval")
    export_animation_parser.add_argument("--window-size", type=int, default=5,
                                       help="Window size for trend analysis")
    export_animation_parser.add_argument("--no-trend", action="store_true",
                                       help="Don't include trend lines")
    export_animation_parser.add_argument("--format", choices=["mp4", "gif"], default="mp4",
                                       help="Animation format")
    
    # Configure export command
    configure_parser = subparsers.add_parser("configure", help="Configure export settings",
                                           parents=[export_args, animation_args])
    configure_parser.add_argument("--no-plotlyjs", action="store_true",
                                help="Don't include plotly.js in HTML exports")
    configure_parser.add_argument("--include-mathjax", action="store_true",
                                help="Include MathJax in HTML exports")
    configure_parser.add_argument("--no-full-html", action="store_true",
                                help="Don't include HTML/body tags in HTML exports")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not HAS_ADVANCED_VISUALIZATION:
        logger.error("Advanced visualization components not available.")
        logger.error("Please install the required dependencies:")
        logger.error("pip install plotly pandas scikit-learn")
        return 1
    
    # Execute command
    if args.command == "export":
        return export_visualization(args)
    elif args.command == "export-all":
        return export_all(args)
    elif args.command == "export-animation":
        return export_animation(args)
    elif args.command == "configure":
        return configure_export(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())