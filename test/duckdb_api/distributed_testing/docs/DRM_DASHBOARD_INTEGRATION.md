# Dynamic Resource Management Dashboard Integration

This document provides information about the integration between the Dynamic Resource Management (DRM) Visualization system and the Monitoring Dashboard in the Distributed Testing Framework.

## Overview

The DRM Dashboard Integration provides a web-based interface for monitoring and analyzing resource allocation, utilization, and scaling decisions in the Distributed Testing Framework. It visualizes key metrics from the Dynamic Resource Manager, including:

- Resource utilization across workers (CPU, memory, GPU)
- Scaling history and events
- Resource allocation patterns
- Resource efficiency metrics
- Cloud resource usage

These visualizations help administrators and developers monitor resource usage, identify bottlenecks, and optimize resource allocation for distributed testing workloads.

## Features

- **Resource Utilization Heatmap**: Visualizes resource utilization across all workers as a heatmap
- **Scaling History**: Shows scaling events over time, including scale-up and scale-down decisions
- **Resource Allocation**: Visualizes how resources are allocated across worker nodes
- **Resource Efficiency**: Shows resource utilization efficiency metrics
- **Cloud Resource Usage**: Displays cloud provider resource usage and costs
- **Interactive Dashboard**: Provides a comprehensive view with tabs for different visualization types
- **Auto-refresh**: Configurable automatic refresh intervals
- **Real-time Monitoring**: Option to launch a WebSocket-based real-time dashboard

## Architecture

The integration consists of several components:

1. **DRMVisualization**: Core visualization module that generates visualizations from DRM data
2. **DRMVisualizationIntegration**: Integration layer that bridges DRMVisualization with the Monitoring Dashboard
3. **DRM Dashboard Route**: Web route in the Monitoring Dashboard that handles DRM dashboard requests
4. **DRM Dashboard Template**: HTML template for displaying the visualizations in the dashboard UI

The integration uses a registry-based approach to track and manage visualizations, storing metadata in a JSON file for persistence.

## Installation

The DRM Dashboard Integration is included in the Distributed Testing Framework and requires no additional installation steps beyond the standard framework installation.

## Usage

### Starting the Dashboard with DRM Integration

To start the Monitoring Dashboard with DRM Visualization Integration:

```bash
python run_dashboard_with_drm_visualization.py
```

Options:
- `--host`: Host to bind the server to (default: localhost)
- `--port`: Port to listen on (default: 8080)
- `--no-mock`: Disable mock data generation (use real DynamicResourceManager)

### Running the Example Script

To see a complete demonstration of the DRM visualization system with simulated data:

```bash
python run_drm_visualization_example.py
```

Options:
- `--host`: Dashboard host (default: localhost)
- `--port`: Dashboard port (default: 8080)
- `--duration`: Runtime in minutes (default: 10)
- `--workers`: Initial number of workers (default: 5)
- `--update-interval`: Seconds between updates (default: 10)

This example:
1. Creates a mock DRM with several workers
2. Simulates changing resource utilization patterns
3. Makes scaling decisions based on utilization thresholds
4. Generates visualizations and updates them periodically
5. Starts an interactive dashboard server

### Accessing the Dashboard

1. Start the dashboard server
2. Navigate to `http://localhost:8080/drm-dashboard` in your web browser

### Dashboard Controls

- **Refresh Visualizations**: Manually refresh all visualizations
- **Start/Open Interactive Dashboard**: Launch or open the interactive dashboard with WebSocket-based real-time updates
- **Auto-refresh**: Configure auto-refresh interval (off, 30 seconds, 1 minute, 5 minutes)

### Visualization Types

Switch between different visualization types using the tabs:

- **Dashboard**: Comprehensive view with all visualizations
- **Resource Utilization**: Heatmap of resource utilization across workers
- **Scaling History**: Timeline of scaling events
- **Resource Allocation**: Visualization of resource allocation patterns
- **Resource Efficiency**: Resource utilization efficiency metrics
- **Cloud Resources**: Cloud provider resource usage and costs (if available)

## Integration with Other Components

The DRM Dashboard integrates with:

- **Dynamic Resource Manager**: Gets resource metrics and scaling data
- **Monitoring Dashboard**: Displays visualizations in the web UI
- **Cloud Provider Integration**: Shows cloud resource usage and costs
- **WebSocket Server**: Provides real-time updates for the interactive dashboard

## Development

### Adding New Visualizations

To add a new visualization type:

1. Add a new visualization method in `DRMVisualization` class
2. Update the `update_visualizations` method in `DRMVisualizationIntegration` to include the new visualization
3. Add the visualization to the dashboard template

### Creating Your Own Visualizations

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

### Running Tests

To run the integration tests:

```bash
python -m unittest test_drm_dashboard_integration.py
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

## Troubleshooting

### Common Issues

- **Visualization Not Showing**: Check if the visualization files exist in the output directory and if the static file server is configured correctly
- **Dashboard Server Error**: Verify that the required dependencies (matplotlib, plotly) are installed
- **Auto-refresh Not Working**: Check browser console for JavaScript errors

### Debugging

- Enable logging to see detailed error messages:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```
  
- Check the visualization registry file (`visualization_registry.json`) to ensure visualizations are being properly registered

## Roadmap

Future enhancements planned for the DRM Dashboard Integration:

- Advanced filtering and time range selection
- Customizable dashboard layouts
- Predictive resource analysis
- Anomaly detection in resource usage patterns
- Enhanced cloud cost optimization visualizations
- Export capabilities for visualization data
- Mobile-optimized views

## References

- [Dynamic Resource Management Documentation](DYNAMIC_RESOURCE_MANAGEMENT.md)
- [DRM Visualization README](../DRM_VISUALIZATION_README.md)
- [Visualization Guide](VISUALIZATION_GUIDE.md)
- [Monitoring Dashboard Guide](../MONITORING_DASHBOARD_GUIDE.md)
- [DRM Visualization Implementation Summary](DRM_VISUALIZATION_IMPLEMENTATION_SUMMARY.md)