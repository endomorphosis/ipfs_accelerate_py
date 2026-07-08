"""
Test script for the 3D Visualization Component.

This script demonstrates various capabilities of the 3D Visualization component
including scatter plots, surface plots, animated plots, and clustered plots.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import the 3D visualization component
from data.duckdb.visualization.advanced_visualization.viz_3d import Visualization3D

def main():
    """Demonstrate the 3D visualization component capabilities."""
    print("Testing 3D Visualization Component")
    
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Determine if Plotly is available or we need to use static images
    try:
        import plotly
        has_plotly = True
        print("Plotly is available - creating interactive HTML visualizations")
        file_ext = ".html"
    except ImportError:
        has_plotly = False
        print("Plotly not available - creating static PNG visualizations")
        file_ext = ".png"
    
    # Create a 3D visualization component
    viz = Visualization3D(debug=True)
    
    # 1. Create a basic 3D scatter plot
    print("\n1. Creating basic 3D scatter plot...")
    scatter_path = output_dir / f"3d_scatter{file_ext}"
    
    viz.create_3d_visualization(
        x_metric="throughput",
        y_metric="memory",
        z_metric="latency",
        color_by="hardware_type",
        model_families=["Text", "Vision"],
        hardware_types=["CPU", "GPU", "WebGPU"],
        output_path=str(scatter_path),
        title="Hardware Performance Comparison: Throughput vs Memory vs Latency"
    )
    print(f"  - Saved to {scatter_path}")
    
    # 2. Create a 3D scatter plot with sizing by batch size
    print("\n2. Creating 3D scatter plot with sized points...")
    sized_scatter_path = output_dir / f"3d_scatter_sized{file_ext}"
    
    viz.create_3d_visualization(
        x_metric="throughput",
        y_metric="memory",
        z_metric="latency",
        color_by="hardware_type",
        size_by="batch_size",
        model_families=["Text", "Vision"],
        hardware_types=["CPU", "GPU", "WebGPU"],
        output_path=str(sized_scatter_path),
        title="Hardware Performance Comparison with Batch Size Scaling"
    )
    print(f"  - Saved to {sized_scatter_path}")
    
    # 3. Create a 3D surface plot
    print("\n3. Creating 3D surface plot...")
    surface_path = output_dir / f"3d_surface{file_ext}"
    
    viz.create_3d_visualization(
        x_metric="throughput",
        y_metric="batch_size",
        z_metric="latency",
        color_by="hardware_type",
        model_families=["Text"],
        hardware_types=["GPU"],
        filter_dict={"model_name": "BERT"},
        output_path=str(surface_path),
        title="Performance Surface: Throughput vs Batch Size vs Latency",
        show_surface=True,
        surface_contours=True
    )
    print(f"  - Saved to {surface_path}")
    
    # 4. Create a 3D plot with regression plane
    print("\n4. Creating 3D plot with regression plane...")
    regression_path = output_dir / f"3d_regression{file_ext}"
    
    viz.create_3d_visualization(
        x_metric="throughput",
        y_metric="memory",
        z_metric="latency",
        color_by="model_family",
        model_families=["Text", "Vision", "Audio"],
        hardware_types=["GPU"],
        output_path=str(regression_path),
        title="Performance with Regression Plane",
        regression_plane=True
    )
    print(f"  - Saved to {regression_path}")
    
    # 5. Create an animated 3D plot (only for HTML)
    if has_plotly:
        print("\n5. Creating animated 3D plot...")
        animated_path = output_dir / "3d_animated.html"
        
        viz.create_3d_visualization(
            x_metric="throughput",
            y_metric="memory",
            z_metric="latency",
            color_by="model_family",
            model_families=["Text", "Vision", "Audio"],
            hardware_types=["CPU", "GPU", "WebGPU"],
            output_path=str(animated_path),
            title="Animated Performance Across Batch Sizes",
            enable_animation=True,
            animation_frame="batch_size",
            animation_speed=0.5
        )
        print(f"  - Saved to {animated_path}")
    
    # 6. Create a 3D plot with clustering
    print("\n6. Creating 3D plot with clustering...")
    clustered_path = output_dir / f"3d_clustered{file_ext}"
    
    viz.create_3d_visualization(
        x_metric="throughput",
        y_metric="memory",
        z_metric="latency",
        model_families=["Text", "Vision", "Audio"],
        hardware_types=["CPU", "GPU", "WebGPU", "WebNN", "MPS"],
        output_path=str(clustered_path),
        title="Clustered Performance Analysis",
        cluster_points=True,
        num_clusters=4,
        show_cluster_centroids=True
    )
    print(f"  - Saved to {clustered_path}")
    
    # 7. Create a 3D plot with projections
    print("\n7. Creating 3D plot with projections...")
    projection_path = output_dir / f"3d_projections{file_ext}"
    
    viz.create_3d_visualization(
        x_metric="throughput",
        y_metric="memory",
        z_metric="latency",
        color_by="hardware_type",
        model_families=["Text"],
        hardware_types=["CPU", "GPU", "WebGPU"],
        output_path=str(projection_path),
        title="Performance with Wall Projections",
        show_projections=True
    )
    print(f"  - Saved to {projection_path}")
    
    # 8. Create a 3D plot with auto-rotation (only for HTML)
    if has_plotly:
        print("\n8. Creating 3D plot with auto-rotation...")
        rotation_path = output_dir / "3d_rotation.html"
        
        viz.create_3d_visualization(
            x_metric="throughput",
            y_metric="memory",
            z_metric="latency",
            color_by="model_name",
            model_families=["Text", "Vision"],
            hardware_types=["GPU"],
            output_path=str(rotation_path),
            title="Auto-Rotating Performance Visualization",
            auto_rotate=True
        )
        print(f"  - Saved to {rotation_path}")
    
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()