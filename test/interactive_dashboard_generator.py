#!/usr/bin/env python3
"""
Interactive Dashboard Generator for the IPFS Accelerate Framework

This script provides an interactive command-line interface for creating custom 
dashboards using the CustomizableDashboard system without writing code.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

try:
    from duckdb_api.visualization.advanced_visualization import CustomizableDashboard
    HAS_DASHBOARD = True
except ImportError:
    print("Error: CustomizableDashboard is not available.")
    print("Make sure you have the required dependencies installed:")
    print("pip install plotly pandas numpy scikit-learn matplotlib")
    HAS_DASHBOARD = False

def main():
    """Main function to run the interactive dashboard generator."""
    parser = argparse.ArgumentParser(description="Interactive Dashboard Generator")
    parser.add_argument("--output-dir", default="./dashboards", help="Output directory")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Dashboard theme")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if not HAS_DASHBOARD:
        sys.exit(1)
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(theme=args.theme, output_dir=args.output_dir, debug=args.debug)
    
    # Interactive prompts
    print("\n=== Interactive Dashboard Generator ===")
    print("=" * 50)
    
    # Get dashboard name and title
    dashboard_name = input("Enter dashboard name: ")
    title = input("Enter dashboard title: ")
    description = input("Enter dashboard description: ")
    
    # Choose template or custom
    use_template = input("Use a template? (y/n): ").lower() == 'y'
    
    if use_template:
        # Show available templates
        templates = dashboard.list_available_templates()
        print("\nAvailable templates:")
        for i, (name, details) in enumerate(templates.items()):
            print(f"{i+1}. {name}: {details.get('title', '')}")
        
        # Select template
        template_idx = int(input("\nSelect template number: ")) - 1
        template_name = list(templates.keys())[template_idx]
        
        # Create dashboard from template
        dashboard_path = dashboard.create_dashboard(
            dashboard_name=dashboard_name,
            template=template_name,
            title=title,
            description=description
        )
    else:
        # Custom dashboard
        components = []
        
        # Add components
        while True:
            add_component = input("\nAdd a component? (y/n): ").lower() == 'y'
            if not add_component:
                break
            
            # Show available component types
            comp_types = dashboard.list_available_components()
            print("\nAvailable component types:")
            for i, (name, desc) in enumerate(comp_types.items()):
                print(f"{i+1}. {name}: {desc}")
            
            # Select component type
            comp_idx = int(input("\nSelect component type number: ")) - 1
            comp_type = list(comp_types.keys())[comp_idx]
            
            # Configure component
            print(f"\nConfiguring {comp_type} component:")
            config = {}
            config["title"] = input("Component title: ")
            
            # Basic configuration based on component type
            if comp_type == "3d":
                metrics_str = input("Metrics (comma-separated, e.g. throughput,latency,memory): ")
                config["metrics"] = [m.strip() for m in metrics_str.split(",")]
                
                dimensions_str = input("Dimensions (comma-separated, e.g. model_family,hardware_type): ")
                config["dimensions"] = [d.strip() for d in dimensions_str.split(",")]
                
            elif comp_type in ["heatmap", "time-series", "animated-time-series"]:
                config["metric"] = input("Metric (e.g. throughput_items_per_second): ")
                
                if comp_type in ["time-series", "animated-time-series"]:
                    dimensions_str = input("Dimensions (comma-separated, e.g. model_family,hardware_type): ")
                    if dimensions_str:
                        config["dimensions"] = [d.strip() for d in dimensions_str.split(",")]
                    
                    time_range = input("Time range in days (e.g. 90): ")
                    if time_range:
                        config["time_range"] = int(time_range)
                    
                    include_trend = input("Include trend line? (y/n): ").lower() == 'y'
                    if include_trend:
                        config["show_trend"] = True
                        window_size = input("Trend window size (e.g. 5): ")
                        if window_size:
                            config["trend_window"] = int(window_size)
                
                if comp_type == "heatmap":
                    model_families = input("Model families (comma-separated, leave empty for all): ")
                    if model_families:
                        config["model_families"] = [m.strip() for m in model_families.split(",")]
                    
                    hardware_types = input("Hardware types (comma-separated, leave empty for all): ")
                    if hardware_types:
                        config["hardware_types"] = [h.strip() for h in hardware_types.split(",")]
            
            # Component size
            width = int(input("\nComponent width (columns): "))
            height = int(input("Component height (rows): "))
            
            # Add component to list
            components.append({
                "type": comp_type,
                "config": config,
                "width": width,
                "height": height
            })
        
        # Get layout settings
        print("\nConfigure dashboard layout:")
        columns = int(input("Number of columns in the grid: "))
        row_height = int(input("Row height in pixels: "))
        
        # Create custom dashboard
        dashboard_path = dashboard.create_dashboard(
            dashboard_name=dashboard_name,
            title=title,
            description=description,
            components=components,
            columns=columns,
            row_height=row_height
        )
    
    if dashboard_path:
        print(f"\nDashboard created successfully: {dashboard_path}")
        
        # Open in browser
        open_browser = input("Open dashboard in browser? (y/n): ").lower() == 'y'
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                print("Dashboard opened in browser.")
            except Exception as e:
                print(f"Error opening dashboard in browser: {e}")
    else:
        print("\nFailed to create dashboard.")

if __name__ == "__main__":
    main()