# Monitoring Dashboard Integration Guide

This guide explains how to use the integrated Monitoring Dashboard with the Advanced Visualization System, including the new Regression Detection capabilities. The integration enables centralized visualization management, embedded dashboards, statistical regression analysis, and improved collaboration.

## Overview

The Monitoring Dashboard Integration connects the Advanced Visualization System to the Distributed Testing Framework's Monitoring Dashboard, allowing users to create, manage, and view customized dashboards with advanced analytics directly within the monitoring interface. This integration provides the following capabilities:

- Create and manage embedded dashboards within the Monitoring Dashboard
- Generate dashboards automatically from monitoring data
- Organize visualizations with a flexible grid layout system
- Perform statistical regression detection with significance testing
- Analyze correlations between different performance metrics
- Create interactive visualizations of performance trends and regressions
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
pip install plotly pandas scikit-learn aiohttp jinja2 dash dash-bootstrap-components scipy ruptures
```

## Running the Monitoring Dashboard with Visualization Integration

The easiest way to use the integration is to run the Monitoring Dashboard with the visualization integration enabled:

```bash
# Run with visualization integration enabled
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration

# Use a specific dashboard directory
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --dashboard-dir ./custom_dashboards

# Enable additional integrations with regression detection
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --enable-regression-detection --enable-performance-analytics
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
5. **Analyze Regressions**: Perform statistical regression analysis on performance data
6. **View Correlations**: Analyze correlations between different metrics

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

# Generate a dashboard with regression detection
dashboard_path = viz_integration.generate_dashboard_with_regression_detection(
    performance_data=analytics_data,
    name="regression_dashboard",
    title="Performance Regression Analysis Dashboard",
    metrics=["latency_ms", "throughput_items_per_second", "memory_usage_mb"]
)

# Get HTML for embedding the dashboard in a web page
iframe_html = viz_integration.get_dashboard_iframe_html(
    name="regression_dashboard",
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
    name="regression_dashboard",
    format="html"
)
```

## Command-Line Tools

### Running the Monitoring Dashboard

To start the Monitoring Dashboard with visualization integration and regression detection:

```bash
# Basic usage with visualization integration and regression detection
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --enable-regression-detection

# Specify dashboard directory
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --enable-regression-detection --dashboard-dir ./dashboards

# Run with additional options
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --enable-regression-detection --port 8082 --theme dark
```

Command-line options:

```
--host                   Host to bind the server to (default: localhost)
--port                   Port to bind the server to (default: 8082)
--dashboard-dir          Directory to store visualization dashboards (default: ./dashboards)
--theme                  Dashboard theme (light or dark, default: dark)
--enable-visualization-integration   Enable visualization integration
--enable-regression-detection       Enable regression detection and analysis
--enable-performance-analytics      Enable performance analytics
--debug                  Enable debug logging
--browser                Open dashboard in browser after starting
```

### Running the Enhanced Visualization Dashboard

For running the stand-alone Enhanced Visualization Dashboard with regression detection:

```bash
# Run with default settings
python run_enhanced_visualization_dashboard.py

# Run with specific options
python run_enhanced_visualization_dashboard.py --port 8085 --theme light --browser

# Run with DB connection
python run_enhanced_visualization_dashboard.py --db-path benchmark_db.duckdb
```

Command-line options:

```
--host                   Host to bind the server to (default: localhost)
--port                   Port to bind the server to (default: 8082)
--db-path                Path to DuckDB database (default: benchmark_db.duckdb)
--output-dir             Output directory for visualizations (default: ./visualizations/dashboard)
--theme                  Dashboard theme (light or dark, default: dark)
--debug                  Enable debug mode
--browser                Open dashboard in browser after starting
```

## Regression Detection Features

The Monitoring Dashboard now includes advanced regression detection capabilities:

### Running Regression Analysis

To run regression analysis on performance data:

1. Navigate to the "Performance Analysis" section in the dashboard
2. Select the "Regression Detection" tab
3. Choose a metric to analyze from the dropdown (e.g., latency, throughput)
4. Click the "Run Regression Analysis" button
5. View the detected regressions with statistical significance highlighted
6. Examine regression details in the sidebar, including severity and significance

### Interpreting Regression Results

The regression detection results include:

- **Time Series Visualization**: Shows the performance data with regression points highlighted
- **Change Points**: Vertical lines indicating where significant changes were detected
- **Regression Annotations**: Shows the percentage change with direction indicators
- **Regression Details**: Provides information about each detected regression:
  - **Severity**: Critical, high, medium, or low based on impact
  - **Statistical Significance**: Confidence level in the detected change
  - **Before/After Values**: Metric values before and after the change point
  - **Percentage Change**: The magnitude of the change

### Correlation Analysis

To analyze correlations between metrics:

1. Navigate to the "Performance Analysis" section
2. Select the "Regression Detection" tab
3. Scroll down to the "Metric Correlations" section
4. Click the "Run Correlation Analysis" button
5. View the correlation matrix heatmap showing relationships between metrics

## Integration with the Distributed Testing Framework

The Monitoring Dashboard Integration is now fully integrated with the Distributed Testing Framework, allowing visualization dashboards with regression detection to be embedded directly within the monitoring interface.

The integration automatically generates dashboards from different types of monitoring data:

1. **Results Page**: Generates dashboards from result aggregator data, showing performance metrics across different models and hardware platforms.

2. **Performance Analytics Page**: Creates dashboards from performance analytics data, visualizing trends, regressions, and patterns in performance metrics over time.

3. **Regression Analysis Page**: Generates dashboards specifically for regression detection and analysis, showing statistically significant performance changes.

To enable the full integration with all features:

```bash
python -m duckdb_api.distributed_testing.run_monitoring_dashboard \
    --enable-visualization-integration \
    --enable-regression-detection \
    --enable-performance-analytics
```

## Advanced Configuration

### Custom Regression Detection Parameters

You can customize regression detection parameters:

```python
# Create a custom regression detector with specific parameters
viz_integration.create_regression_detection_dashboard(
    name="custom_regression_dashboard",
    title="Custom Regression Analysis",
    regression_params={
        "min_samples": 10,                # Minimum samples for detection
        "window_size": 15,                # Window size for moving average
        "regression_threshold": 5.0,      # Percentage change threshold
        "confidence_level": 0.99,         # Statistical confidence level
        "smoothing_factor": 0.3,          # Time series smoothing factor
        "allow_positive_regressions": True, # Include improvements
        "severity_thresholds": {
            "critical": 25.0,
            "high": 15.0,
            "medium": 8.0,
            "low": 3.0
        }
    },
    metrics=["latency_ms", "throughput_items_per_second"]
)
```

### Custom Correlation Analysis

For specialized correlation analysis:

```python
# Create a custom correlation analysis dashboard
viz_integration.create_correlation_analysis_dashboard(
    name="custom_correlation_dashboard",
    title="Custom Correlation Analysis",
    correlation_params={
        "correlation_threshold": 0.7,     # Minimum correlation to highlight
        "correlation_method": "pearson",  # Correlation method (pearson, spearman, kendall)
        "include_p_values": True,         # Include p-values in the visualization
        "cluster_metrics": True           # Group similar metrics together
    },
    metrics=["latency_ms", "throughput_items_per_second", "memory_usage_mb", "cpu_usage", "gpu_usage"]
)
```

## Testing the Integration

### Running Integration Tests

You can run the integration tests with the provided test runner script:

```bash
# Run all integration tests including regression detection
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --include-regression-tests

# Run regression detection tests specifically
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --test regression_detection

# Create dummy data for regression testing
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --create-regression-test-data
```

### Integration Test Components

The integration tests now include these additional components:

1. **Regression Detection Tests** (`test_regression_detection.py`):
   - Tests the `RegressionDetector` class functionality
   - Validates statistical significance testing
   - Tests change point detection algorithms
   - Ensures visualization generation works properly

2. **Correlation Analysis Tests** (`test_correlation_analysis.py`):
   - Tests correlation matrix generation
   - Validates correlation insight detection
   - Tests correlation visualization creation

3. **Dashboard Integration Tests** (`test_dashboard_regression_integration.py`):
   - Tests integration of regression detection with the dashboard
   - Validates UI components for regression analysis
   - Tests end-to-end regression detection workflow

## Best Practices

1. **Set Appropriate Thresholds**: Configure regression thresholds based on your application's sensitivity to performance changes
2. **Use Statistical Significance**: Focus on statistically significant regressions to reduce false positives
3. **Analyze Correlations**: Use correlation analysis to identify relationships between metrics that can help diagnose issues
4. **Combine Analyses**: Use regression detection alongside other analyses for a comprehensive view
5. **Create Focused Dashboards**: Create specific dashboards for different types of analysis
6. **Add Context**: Include detailed descriptions and metadata for better interpretation
7. **Regular Monitoring**: Set up automated regression detection to catch issues early

## Conclusion

The Monitoring Dashboard Integration with enhanced regression detection provides powerful tools for visualizing and analyzing performance data. The statistical regression detection capabilities enable teams to identify significant performance changes with confidence, while the correlation analysis helps understand relationships between different metrics.

For further information, see:
- [ADVANCED_VISUALIZATION_GUIDE.md](ADVANCED_VISUALIZATION_GUIDE.md) - Comprehensive guide to the visualization system
- [MONITORING_DASHBOARD_GUIDE.md](duckdb_api/distributed_testing/docs/MONITORING_DASHBOARD_GUIDE.md) - Guide for the monitoring dashboard
- [PERFORMANCE_DASHBOARD_SPECIFICATION.md](PERFORMANCE_DASHBOARD_SPECIFICATION.md) - Technical specification for the performance dashboard
- [VISUALIZATION_DASHBOARD_README.md](VISUALIZATION_DASHBOARD_README.md) - README for the enhanced visualization dashboard