# Dynamic Resource Management Visualization Guide

This guide provides an overview of the Dynamic Resource Management (DRM) visualization capabilities in the Distributed Testing Framework.

## Table of Contents

1. [Overview](#overview)
2. [Visualization Types](#visualization-types)
3. [Integration with DRM](#integration-with-drm)
4. [Running the Dashboard](#running-the-dashboard)
5. [Monitoring Dashboard Integration](#monitoring-dashboard-integration)
6. [Integration Testing](#integration-testing)
7. [API Reference](#api-reference)

## Overview

The DRM Visualization system provides comprehensive visualization capabilities for monitoring and analyzing the Dynamic Resource Management component of the Distributed Testing Framework. It helps in understanding resource allocation patterns, scaling decisions, and resource utilization across worker nodes.

Key features:
- Resource utilization heatmaps
- Scaling history visualizations
- Resource allocation and efficiency metrics
- Cloud resource usage tracking
- Interactive web dashboard with real-time updates

## Visualization Types

### Resource Utilization Heatmap

This visualization shows resource utilization (CPU, memory, GPU) across all workers over time, making it easy to identify patterns and potential bottlenecks.

![Resource Utilization Heatmap](../docs/images/resource_utilization_heatmap_example.png)

### Scaling History Visualization

This visualization tracks scaling decisions (scale up, scale down, maintain) over time alongside resource utilization, helping to understand the effectiveness of scaling policies.

![Scaling History](../docs/images/scaling_history_example.png)

### Resource Allocation Visualization

This visualization shows how different resources (CPU, memory, GPU) are allocated across worker nodes, helping to identify imbalances or optimization opportunities.

![Resource Allocation](../docs/images/resource_allocation_example.png)

### Resource Efficiency Visualization

This visualization measures how efficiently allocated resources are being utilized, identifying potential over-provisioning or under-provisioning.

![Resource Efficiency](../docs/images/resource_efficiency_example.png)

### Cloud Resource Visualization

If cloud provider integration is available, this visualization tracks resource usage and costs across different cloud providers.

![Cloud Resources](../docs/images/cloud_resources_example.png)

## Integration with DRM

The visualization system integrates directly with the Dynamic Resource Manager, automatically collecting data on:
- Worker resource allocation and utilization
- Task assignment and execution
- Scaling decisions
- Cloud provider usage

### Basic Usage

```python
from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager
from duckdb_api.distributed_testing.dynamic_resource_management_visualization import DRMVisualization

# Create a DRM instance
drm = DynamicResourceManager()

# Create a visualization instance
visualization = DRMVisualization(
    dynamic_resource_manager=drm,
    output_dir="./visualizations",
    interactive=True  # Use interactive Plotly visualizations
)

# Generate visualizations
heatmap_path = visualization.create_resource_utilization_heatmap()
scaling_path = visualization.create_scaling_history_visualization()
dashboard_path = visualization.create_resource_dashboard()

# Start a web dashboard server
dashboard_url = visualization.start_dashboard_server(port=8889)
print(f"Dashboard available at: {dashboard_url}")

# When done
visualization.cleanup()
```

## Running the Dashboard

The visualization system includes a real-time web dashboard that updates as new data becomes available:

1. Create a visualization instance with your DRM
2. Start the dashboard server
3. Access the dashboard at the provided URL

```python
# Start the dashboard server
url = visualization.start_dashboard_server(port=8889)
print(f"Dashboard available at: {url}")

# Keep the server running
try:
    # Run your workload here...
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Stop the server when done
    visualization.stop_dashboard_server()
```

## Monitoring Dashboard Integration

The DRM Visualization system integrates with the Monitoring Dashboard to provide a comprehensive view of resource management within the distributed testing framework.

### Accessing the DRM Dashboard

The DRM Dashboard is accessible via the monitoring dashboard at the following URL:

```
http://<dashboard-host>:<dashboard-port>/drm-dashboard
```

For example, if the monitoring dashboard is running on localhost:8080, you can access the DRM dashboard at:

```
http://localhost:8080/drm-dashboard
```

### Dashboard Features

The DRM Dashboard provides the following features:

- **Resource Summary**: Overview of active workers, total resources, utilization, and scaling events
- **Tabbed Interface**: Easy navigation between different visualization types
- **Resource Utilization**: Heatmap showing resource utilization across workers
- **Scaling History**: Timeline of scaling decisions and their impact
- **Resource Allocation**: Visualization of resource distribution across workers
- **Resource Efficiency**: Metrics showing efficiency of resource allocation
- **Cloud Resources**: Tracking of cloud provider resource usage (if available)
- **Interactive Dashboard**: Option to launch a separate interactive dashboard with WebSocket-based real-time updates
- **Auto-refresh**: Configurable auto-refresh intervals for real-time monitoring

### Running the DRM Dashboard with the Monitoring Dashboard

To run the monitoring dashboard with DRM visualization integration:

```bash
cd duckdb_api/distributed_testing
python run_dashboard_with_drm_visualization.py
```

Options:
- `--host HOST`: Host to bind the server to (default: localhost)
- `--port PORT`: Port to listen on (default: 8080)
- `--no-mock`: Disable mock data generation (use real DynamicResourceManager)

### Programmatic Integration

You can programmatically integrate the DRM visualization with the monitoring dashboard:

```python
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
from duckdb_api.distributed_testing.dashboard.drm_visualization_integration import DRMVisualizationIntegration
from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager

# Create a DRM instance
drm = DynamicResourceManager()

# Create DRM visualization integration
drm_integration = DRMVisualizationIntegration(
    output_dir="./drm_visualizations",
    update_interval=60,  # 1 minute update interval
    resource_manager=drm
)

# Create and start the monitoring dashboard with DRM integration
dashboard = MonitoringDashboard(
    host="localhost",
    port=8080,
    dashboard_dir="./dashboards"
)

# Set the DRM integration
dashboard.drm_visualization_integration = drm_integration

# Start the dashboard
dashboard.run()
```

## Integration Testing

The repository includes integration tests for the visualization system that demonstrate its capabilities with simulated data.

### Running the Tests

```bash
cd duckdb_api/distributed_testing
python -m tests.test_dynamic_resource_management_visualization
```

### Testing Dashboard Integration

To test the integration between DRM visualization and the monitoring dashboard:

```bash
cd duckdb_api/distributed_testing
python -m tests.test_drm_dashboard_integration
```

### Running the Integration Demo

The repository includes a comprehensive integration demo that simulates a dynamic workload and generates all visualization types:

```bash
cd duckdb_api/distributed_testing
python -m tests.run_drm_visualization_integration_test --interactive --dashboard
```

Options:
- `--interactive`: Use interactive Plotly visualizations
- `--dashboard`: Start the web dashboard server
- `--output-dir PATH`: Specify output directory (default: ./visualization_output)
- `--duration SECONDS`: Specify workload simulation duration (default: 30)

## API Reference

### DRMVisualization

The main class that provides visualization capabilities.

#### Constructor

```python
DRMVisualization(
    dynamic_resource_manager=None,      # DynamicResourceManager instance
    cloud_provider_manager=None,        # Optional CloudProviderManager
    resource_optimizer=None,            # Optional ResourceOptimizer
    output_dir=None,                    # Directory for output files
    dashboard_port=8889,                # Port for web dashboard
    data_retention_days=30,             # Days of history to keep
    update_interval=300,                # Seconds between data updates
    interactive=True                    # Use interactive visualizations
)
```

#### Methods

- `create_resource_utilization_heatmap(output_path=None, show_plot=False)`: Create a resource utilization heatmap
- `create_scaling_history_visualization(output_path=None, show_plot=False)`: Create a scaling history visualization
- `create_resource_allocation_visualization(output_path=None, show_plot=False)`: Create a resource allocation visualization
- `create_resource_efficiency_visualization(output_path=None, show_plot=False)`: Create a resource efficiency visualization
- `create_cloud_resource_visualization(output_path=None, show_plot=False)`: Create a cloud resource usage visualization
- `create_resource_dashboard(output_dir=None)`: Create a comprehensive dashboard with all visualizations
- `start_dashboard_server(port=None, background=True)`: Start a web server for the dashboard
- `stop_dashboard_server()`: Stop the dashboard server
- `cleanup()`: Clean up resources used by the visualization system