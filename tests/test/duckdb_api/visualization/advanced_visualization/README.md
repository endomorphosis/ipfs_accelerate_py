# Advanced Visualization System

This directory contains the Advanced Visualization System for the IPFS Accelerate Python Framework. It provides comprehensive visualization capabilities for model performance data across different hardware platforms, batch sizes, and precision formats.

## Key Components

1. **BaseVisualization**: Base class for all visualization components
2. **Visualization3D**: Interactive 3D visualization for multi-dimensional data
3. **HardwareHeatmapVisualization**: Heatmaps for comparing performance across hardware platforms
4. **TimeSeriesVisualization**: Time-series plots for tracking metrics over time
5. **AnimatedTimeSeriesVisualization**: Animated time-series visualizations with interactive controls
6. **CustomizableDashboard**: Customizable dashboard system for combining multiple visualizations

## Integration with Monitoring Dashboard

The Advanced Visualization System can now be integrated with the Monitoring Dashboard through the following components:

- **VisualizationDashboardIntegration**: Provides integration between the Customizable Dashboard and Monitoring Dashboard
- **Dashboard Management UI**: Web interface for managing embedded dashboards within the monitoring dashboard
- **Automatic Dashboard Generation**: Creates dashboards from monitoring data with appropriate visualizations

## Using the Visualization Components

### 3D Visualization

```python
from duckdb_api.visualization.advanced_visualization import Visualization3D

# Create a 3D visualization component
viz_3d = Visualization3D(theme="light")

# Create a 3D visualization
viz_3d.create_3d_visualization(
    metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
    dimensions=["model_family", "hardware_type"],
    title="3D Performance Visualization"
)
```

### Hardware Heatmap

```python
from duckdb_api.visualization.advanced_visualization import HardwareHeatmapVisualization

# Create a heatmap visualization component
viz_heatmap = HardwareHeatmapVisualization(theme="light")

# Create a hardware comparison heatmap
viz_heatmap.create_hardware_heatmap(
    metric="throughput_items_per_second",
    model_families=["transformers", "vision", "audio"],
    hardware_types=["nvidia_a100", "amd_mi250", "intel_arc"],
    title="Hardware Throughput Comparison"
)
```

### Time-Series Visualization

```python
from duckdb_api.visualization.advanced_visualization import TimeSeriesVisualization

# Create a time-series visualization component
viz_ts = TimeSeriesVisualization(theme="light")

# Create a time-series visualization
viz_ts.create_time_series_visualization(
    metric="throughput_items_per_second",
    dimensions=["model_family", "hardware_type"],
    time_range=90,
    include_trend=True,
    title="Performance Trends Over Time"
)
```

### Animated Time-Series Visualization

```python
from duckdb_api.visualization.advanced_visualization import AnimatedTimeSeriesVisualization

# Create an animated time-series visualization component
viz_animated = AnimatedTimeSeriesVisualization(theme="light")

# Create an animated time-series visualization
viz_animated.create_animated_time_series(
    metric="throughput_items_per_second",
    dimensions=["model_family", "hardware_type"],
    time_range=90,
    show_trend=True,
    title="Animated Performance Trends"
)
```

### Customizable Dashboard

```python
from duckdb_api.visualization.advanced_visualization import CustomizableDashboard

# Create a dashboard component
dashboard = CustomizableDashboard(theme="light", output_dir="./dashboards")

# Create a dashboard from a template
dashboard_path = dashboard.create_dashboard(
    dashboard_name="performance_overview",
    template="overview",
    title="Performance Overview Dashboard",
    description="Overview of performance metrics across models and hardware"
)

# Add a component to the dashboard
dashboard.add_component_to_dashboard(
    dashboard_name="performance_overview",
    component_type="heatmap",
    component_config={
        "metric": "memory_peak_mb",
        "title": "Memory Usage Comparison"
    },
    width=2,
    height=1
)
```

## Command-Line Interfaces

The Advanced Visualization System provides several command-line interfaces for easy access:

### Running the Customizable Dashboard

```bash
# Create a dashboard from a template
python run_customizable_dashboard.py create --name hardware_overview --template hardware_comparison --open

# Create a custom dashboard with all visualization types
python run_customizable_dashboard.py create --name custom_dashboard --include-all --open

# List all saved dashboards
python run_customizable_dashboard.py list

# Add a component to a dashboard
python run_customizable_dashboard.py add --name hardware_overview --type heatmap --config-file config.json --open
```

### Testing the Advanced Visualization System

```bash
# Test the customizable dashboard with all templates
python run_dashboard_tests.py --template all

# Test dashboard management features
python run_dashboard_tests.py --management-test

# Test the animated time-series visualization
python test_animated_time_series.py

# Test the 3D visualization
python test_3d_visualization.py
```

### Monitoring Dashboard Integration

```python
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import VisualizationDashboardIntegration

# Create integration component
viz_integration = VisualizationDashboardIntegration(
    dashboard_dir="./dashboards",
    integration_dir="./dashboards/monitor_integration"
)

# Create an embedded dashboard for the overview page
viz_integration.create_embedded_dashboard(
    name="overview_dashboard",
    page="index",
    template="overview",
    title="System Overview Dashboard",
    description="Overview of system metrics",
    position="below"
)

# Generate a dashboard from performance data
viz_integration.generate_dashboard_from_performance_data(
    performance_data=analytics_data,
    name="performance_dashboard",
    title="Performance Analytics Dashboard"
)

# Get HTML for embedding a dashboard
iframe_html = viz_integration.get_dashboard_iframe_html(
    name="overview_dashboard",
    width="100%",
    height="600px"
)
```

### Running the Monitoring Dashboard with Visualization Integration

```bash
# Run the monitoring dashboard with visualization integration enabled
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration

# Run with specific dashboard directory
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --dashboard-dir ./custom_dashboards
```

## Documentation

For comprehensive documentation, see:
- [ADVANCED_VISUALIZATION_GUIDE.md](../../../../ADVANCED_VISUALIZATION_GUIDE.md): Complete guide to the Advanced Visualization System
- [WEB_RESOURCE_POOL_DOCUMENTATION.md](../../../../WEB_RESOURCE_POOL_DOCUMENTATION.md): Documentation for web resource pool with visualization integration

## Dependencies

- Plotly: Interactive visualizations
- Matplotlib: Static visualizations
- Pandas: Data handling
- NumPy: Numerical computations
- scikit-learn: Statistical analysis and machine learning
- aiohttp: For web server integration

To install all dependencies:

```bash
pip install plotly pandas numpy scikit-learn matplotlib aiohttp
```