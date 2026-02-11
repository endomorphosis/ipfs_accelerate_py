# Dynamic Resource Management Visualization

This directory contains the Dynamic Resource Management (DRM) visualization system for the Distributed Testing Framework. The DRM visualization system provides comprehensive visualizations of resource allocation, utilization, scaling decisions, and efficiency metrics for the distributed worker nodes.

## Overview

The DRM visualization system offers several types of visualizations:

1. **Resource Utilization Heatmaps**: Visualizes CPU, memory, and GPU utilization across workers over time
2. **Scaling History**: Tracks scaling decisions (scale up, scale down, maintain) and their impact
3. **Resource Allocation**: Shows how different resource types are distributed across workers
4. **Resource Efficiency**: Measures how efficiently allocated resources are being utilized
5. **Cloud Resource Usage**: Tracks resource usage and costs across cloud providers

These visualizations are available as both static images (PNG) and interactive dashboards (HTML).

## Components

The DRM visualization system consists of the following components:

- `dynamic_resource_manager.py`: Core implementation of the DRM system
- `dynamic_resource_management_visualization.py`: Visualization module for DRM
- `dashboard/drm_visualization_integration.py`: Integration with the monitoring dashboard
- `dashboard/templates/drm_dashboard.html`: Dashboard template
- `run_drm_visualization_example.py`: Example script demonstrating the visualization system

## Usage

### Running the Example

To see the visualization system in action with simulated data:

```bash
python run_drm_visualization_example.py
```

This will:
1. Create a mock DRM with several workers
2. Simulate changing resource utilization patterns
3. Make scaling decisions based on utilization thresholds
4. Generate visualizations and update them periodically
5. Start an interactive dashboard server

### Command-line Options

The example script supports several command-line options:

- `--host`: Dashboard host (default: localhost)
- `--port`: Dashboard port (default: 8080)
- `--duration`: Runtime in minutes (default: 10)
- `--workers`: Initial number of workers (default: 5)
- `--update-interval`: Seconds between updates (default: 10)

Example with custom settings:

```bash
python run_drm_visualization_example.py --port 8888 --duration 30 --workers 10 --update-interval 5
```

## Integration with Monitoring Dashboard

The DRM visualization system integrates with the main monitoring dashboard, allowing you to:

1. View DRM visualizations directly in the monitoring dashboard
2. Access visualizations via the sidebar menu
3. Control dashboard updates and refresh intervals
4. Start and stop the interactive dashboard server

To run the monitoring dashboard with DRM visualization:

```bash
python run_dashboard_with_drm_visualization.py
```

## Creating Your Own Visualizations

To use the visualization system with your own DynamicResourceManager:

```python
from dynamic_resource_management_visualization import DRMVisualization

# Create a visualization instance
visualization = DRMVisualization(
    dynamic_resource_manager=your_drm_instance,
    output_dir="./visualizations",
    interactive=True
)

# Generate visualizations
visualization.create_resource_utilization_heatmap()
visualization.create_scaling_history_visualization()
visualization.create_resource_allocation_visualization()
visualization.create_resource_efficiency_visualization()

# Create comprehensive dashboard
dashboard_path = visualization.create_resource_dashboard()

# Start interactive dashboard server
visualization.start_dashboard_server(port=8889)
```

## Dependencies

The visualization system requires the following Python packages:

- matplotlib
- numpy
- pandas

For interactive visualizations:
- plotly

For the dashboard server:
- tornado

## Files and Directories

- `visualization_output/`: Default output directory for visualizations
- `visualization_registry.json`: Registry tracking available visualizations
- Generated visualization files (PNG, HTML)

## Dashboard Features

The DRM dashboard provides several features:

- **Tabbed Interface**: Switch between different visualization types
- **Auto-refresh**: Configure automatic updates (30s, 1m, 5m)
- **Interactive Controls**: Start/stop dashboard server, refresh visualizations
- **Summary Cards**: Quick overview of resource statistics
- **Responsive Design**: Works on different screen sizes

## Troubleshooting

If you encounter issues:

1. Check that dependencies are installed
2. Verify that the DRM instance is properly initialized
3. Check log output for errors
4. Ensure output directories are writable
5. For dashboard server issues, check if the port is already in use