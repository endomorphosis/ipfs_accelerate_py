# Monitoring Dashboard Integration Guide

This guide explains how to use the integrated Monitoring Dashboard with the Advanced Visualization System. The integration enables centralized visualization management, embedded dashboards, and improved collaboration.

## Overview

The Monitoring Dashboard Integration connects the Advanced Visualization System to the Distributed Testing Framework's Monitoring Dashboard, allowing users to create, manage, and view customized dashboards directly within the monitoring interface. This integration provides the following capabilities:

- Create and manage embedded dashboards within the Monitoring Dashboard
- Generate dashboards automatically from monitoring data
- Organize visualizations with a flexible grid layout system
- Manage dashboards through a dedicated Dashboard Management UI
- Export dashboards in various formats (HTML, PNG, PDF)
- Create dashboards from templates or with custom components

## Getting Started

### Prerequisites

To use the Monitoring Dashboard Integration, you need:

- Python 3.8+ with required dependencies installed
- Access to visualization data via the DuckDB API
- The Distributed Testing Framework's Monitoring Dashboard

### Installation

Install the required dependencies:

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install dependencies directly
pip install plotly pandas scikit-learn aiohttp jinja2
```

## Running the Monitoring Dashboard with Visualization Integration

The easiest way to use the integration is to run the Monitoring Dashboard with the visualization integration enabled:

```bash
# Run with visualization integration enabled
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration

# Use a specific dashboard directory
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --dashboard-dir ./custom_dashboards

# Enable additional integrations
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --enable-e2e-test-integration --enable-performance-analytics
```

Once the dashboard is running, you can access it at `http://localhost:8082` by default, and manage dashboards at `http://localhost:8082/dashboards`.

## Basic Usage

There are two ways to use the dashboard integration:

1. **Dashboard Management UI**: Use the web-based management interface for a user-friendly experience
2. **VisualizationDashboardIntegration API**: Use the API for programmatic control

### Dashboard Management UI

The Dashboard Management UI provides a web-based interface for managing dashboards within the Monitoring Dashboard. You can access it at:

```
http://localhost:8082/dashboards
```

From this interface, you can:

1. **List Dashboards**: View all registered dashboards
2. **Create Dashboards**: Create new dashboards from templates or with custom components
3. **Update Dashboards**: Modify dashboard properties and layout
4. **Remove Dashboards**: Delete dashboards from the system

### VisualizationDashboardIntegration API

For programmatic control, you can use the `VisualizationDashboardIntegration` class:

```python
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import VisualizationDashboardIntegration

# Create the integration component
viz_integration = VisualizationDashboardIntegration(
    dashboard_dir="./dashboards",
    integration_dir="./dashboards/monitor_integration"
)

# Create an embedded dashboard for the overview page
dashboard_details = viz_integration.create_embedded_dashboard(
    name="overview_dashboard",
    page="index",
    template="overview",
    title="System Overview Dashboard",
    description="Overview of system performance metrics",
    position="below"  # Can be "above", "below", or "tab"
)

# Generate a dashboard from performance data
dashboard_path = viz_integration.generate_dashboard_from_performance_data(
    performance_data=analytics_data,
    name="performance_dashboard",
    title="Performance Analytics Dashboard"
)

# Get HTML for embedding the dashboard in a web page
iframe_html = viz_integration.get_dashboard_iframe_html(
    name="overview_dashboard",
    width="100%",
    height="600px"
)

# Update an embedded dashboard
viz_integration.update_embedded_dashboard(
    name="overview_dashboard",
    title="Updated Overview Dashboard",
    description="Updated description",
    position="above",
    page="results"
)

# Remove an embedded dashboard
viz_integration.remove_embedded_dashboard("overview_dashboard")

# Get a list of available templates
templates = viz_integration.list_available_templates()

# Get a list of available components
components = viz_integration.list_available_components()

# Export a dashboard to a different format
viz_integration.export_embedded_dashboard(
    name="overview_dashboard",
    format="html"
)
```

## Command-Line Tools

### Running the Monitoring Dashboard

To start the Monitoring Dashboard with visualization integration:

```bash
# Basic usage with visualization integration
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration

# Specify dashboard directory
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --dashboard-dir ./dashboards

# Run with additional options
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --port 8082 --theme dark
```

Command-line options:

```
--host                   Host to bind the server to (default: localhost)
--port                   Port to bind the server to (default: 8082)
--dashboard-dir          Directory to store visualization dashboards (default: ./dashboards)
--theme                  Dashboard theme (light or dark, default: dark)
--enable-visualization-integration   Enable visualization integration
--enable-e2e-test-integration       Enable E2E test integration
--enable-performance-analytics      Enable performance analytics
--debug                  Enable debug logging
--browser                Open dashboard in browser after starting
```

### Running the Customizable Dashboard CLI

The Advanced Visualization System's Customizable Dashboard provides its own CLI for creating and managing dashboards:

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

Command-line options:

```
# Global options
--output-dir       Directory to save dashboards (default: ./dashboards)
--debug            Enable debug mode

# Create command options
--name             Dashboard name
--template         Dashboard template to use (overview, hardware_comparison, model_analysis, empty)
--title            Dashboard title
--description      Dashboard description
--columns          Number of columns in the grid layout
--row-height       Height of each row in pixels
--theme            Dashboard theme (light or dark)
--include-3d       Include 3D visualization component
--include-heatmap  Include heatmap visualization components
--include-timeseries Include time-series visualization component
--include-animated Include animated time-series visualization component
--include-all      Include all visualization components
--open             Open dashboard in browser after creation
```

## Integration with the Distributed Testing Framework

The Monitoring Dashboard Integration is now fully integrated with the Distributed Testing Framework, allowing visualization dashboards to be embedded directly within the monitoring interface.

The integration automatically generates dashboards from different types of monitoring data:

1. **Results Page**: Generates dashboards from result aggregator data, showing performance metrics across different models and hardware platforms.

2. **Performance Analytics Page**: Creates dashboards from performance analytics data, visualizing trends and patterns in performance metrics over time.

3. **E2E Test Results Page**: Generates dashboards from end-to-end test results, showing test execution metrics and validation results.

To enable the full integration with all features:

```bash
python -m duckdb_api.distributed_testing.run_monitoring_dashboard \
    --enable-visualization-integration \
    --enable-e2e-test-integration \
    --enable-performance-analytics
```

Then access the integrated dashboard at:

```
http://localhost:8082
```

And manage dashboards at:

```
http://localhost:8082/dashboards
```

## Troubleshooting

### Dashboard Not Displaying Properly

If embedded dashboards don't display properly:

1. Check if the dashboard directory exists and contains the dashboard files
2. Verify the symbolic link to the dashboard directory in the static directory
3. Make sure the dashboard HTML files are properly formatted
4. Look for error messages in the logs

### Dashboard Management UI Issues

If the Dashboard Management UI doesn't work as expected:

1. Make sure the visualization integration is enabled with `--enable-visualization-integration`
2. Check if the dashboard templates are available
3. Verify that the CustomizableDashboard class is properly initialized
4. Check for JavaScript errors in the browser console

### Missing Dashboard Templates

If dashboard templates are missing:

1. Make sure the Advanced Visualization System is properly installed
2. Check if the dashboard templates are defined in the CustomizableDashboard class
3. Try running the system with the `--debug` flag to see more detailed logs

## Testing the Integration

The Distributed Testing Framework includes comprehensive tests for the Monitoring Dashboard and Advanced Visualization System integration. These tests ensure the components work correctly together and maintain compatibility as the system evolves.

### Running Integration Tests

You can run the integration tests with the provided test runner script:

```bash
# Run all integration tests
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests

# Run with verbose output
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --verbose

# Run a specific test pattern
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --test create_embedded_dashboard

# Create dummy data for testing
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --create-dummy-data
```

### Integration Test Components

The integration tests include the following key components:

1. **Core Integration Tests** (`test_visualization_dashboard_integration.py`):
   - Tests the `VisualizationDashboardIntegration` class
   - Validates creating, updating, and removing embedded dashboards
   - Tests dashboard generation from performance data
   - Ensures template and component listing functionality works

2. **Web Server Integration Tests** (`test_dashboard_visualization_web_integration.py`):
   - Tests serving dashboard files via HTTP
   - Validates iframe HTML generation for embedding
   - Tests route handlers with mock implementations
   - Ensures proper file serving through the web server

3. **Monitoring Dashboard Tests** (`TestMonitoringDashboardWithVisualization`):
   - Tests monitoring dashboard initialization with visualization
   - Validates creating default dashboards
   - Tests embedding dashboards in monitoring pages

These tests provide coverage for all the key functionality of the integration, ensuring that it works reliably in a variety of scenarios.

## Advanced Configuration

### Custom Layout for Dashboard Panels

You can create custom layouts for dashboard panels:

```python
# Create a custom layout with different sizes
layout = {
    "type": "grid",
    "columns": 3,
    "rows": 2,
    "items": [
        # Large 3D visualization spanning 2 columns
        {"visualization_id": "3d_viz", "row": 0, "col": 0, "width": 2, "height": 1},
        # Small heatmap in the top right
        {"visualization_id": "heatmap_viz", "row": 0, "col": 2, "width": 1, "height": 1},
        # Time series at the bottom spanning full width
        {"visualization_id": "time_series_viz", "row": 1, "col": 0, "width": 3, "height": 1}
    ]
}

# Create panel with custom layout
integration.create_dashboard_panel(
    panel_title="Custom Layout Panel",
    visualization_ids=["3d_viz", "heatmap_viz", "time_series_viz"],
    layout=layout
)
```

### Real-Time Updates with WebSocket

For applications requiring real-time updates:

```python
# Connect to WebSocket
integration.connect_websocket()

# Send updates
while True:
    # Generate new data
    new_data = generate_data()
    
    # Send update
    integration.send_real_time_update(
        visualization_id="real_time_viz",
        update_data=new_data
    )
    
    # Wait before next update
    time.sleep(1)
```

## Architecture

The Monitoring Dashboard Integration consists of several components:

1. **MonitorDashboardIntegration**: Core class providing direct communication with the dashboard API
2. **MonitorDashboardIntegrationMixin**: Mixin class that adds dashboard capabilities to visualization systems
3. **DashboardEnhancedVisualizationSystem**: Complete visualization system with dashboard integration
4. **Command-line tools**: Scripts for easy usage from the command line

### API Communication

The integration uses HTTP and WebSocket protocols to communicate with the dashboard:

- HTTP REST API for visualization registration, panel creation, and snapshot management
- WebSocket API for real-time updates to visualizations

### Data Flow

1. Visualizations are created and saved as HTML files
2. Visualization metadata is registered with the dashboard
3. Dashboard panels are created to organize visualizations
4. Users view visualizations through the dashboard interface
5. Real-time updates are sent via WebSocket when data changes
6. Dashboard snapshots capture the state for sharing or backup

## Best Practices

1. **Organize Visualizations**: Use consistent naming and organization for visualizations
2. **Use Real-Time Updates Sparingly**: Only send updates when data significantly changes
3. **Create Focused Panels**: Group related visualizations into panels for better analysis
4. **Use Auto-Sync for Automation**: Set up automatic synchronization for continuous integration
5. **Snapshot for Preservation**: Create snapshots to preserve important analysis states
6. **Follow Naming Conventions**: Use consistent naming for visualizations and panels
7. **Add Metadata**: Include detailed metadata for better searchability and context

## Conclusion

The Monitoring Dashboard Integration provides a powerful way to share and collaborate on visualizations. By centralizing visualizations in a dashboard, teams can more effectively analyze and communicate their findings. The integration is designed to be flexible and easy to use, with both API and command-line interfaces to suit different workflows.

For further information, see:
- [ADVANCED_VISUALIZATION_GUIDE.md](ADVANCED_VISUALIZATION_GUIDE.md) - Comprehensive guide to the visualization system
- [MONITORING_DASHBOARD_GUIDE.md](duckdb_api/distributed_testing/docs/MONITORING_DASHBOARD_GUIDE.md) - Guide for the monitoring dashboard
- [INTEGRATION_EXTENSIBILITY_COMPLETION.md](distributed_testing/docs/INTEGRATION_EXTENSIBILITY_COMPLETION.md) - Report on integration and extensibility components