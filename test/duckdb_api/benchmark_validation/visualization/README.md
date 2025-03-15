# Benchmark Validation Visualization Components

This directory contains the visualization components for the Benchmark Validation System. These components provide comprehensive reporting, visualization, and dashboard capabilities for analyzing benchmark validation results.

## Components

The package includes two primary components:

1. **ValidationReporter**: Generates reports and individual visualizations for validation results.
2. **ValidationDashboard**: Creates comprehensive dashboards with multiple visualizations and integration with the Monitoring Dashboard system.

## ValidationReporter

The `ValidationReporterImpl` class implements the `ValidationReporter` interface and provides comprehensive reporting and visualization capabilities:

### Features

- **Multiple Report Formats**: Generate reports in HTML, Markdown, and JSON formats.
- **Interactive Visualizations**: Create interactive visualizations for validation results.
- **Export Capabilities**: Export reports and visualizations to files.
- **Theming Support**: Customize report appearance with light/dark themes.
- **Advanced Visualization Integration**: Integration with the Advanced Visualization System when available.

### Usage

```python
from duckdb_api.benchmark_validation.visualization import ValidationReporterImpl

# Create a reporter instance
reporter = ValidationReporterImpl({
    "output_directory": "./reports",
    "theme": "light"
})

# Generate HTML report with visualizations
html_report = reporter.generate_report(
    validation_results=validation_results,
    report_format="html",
    include_visualizations=True
)

# Export report to file
reporter.export_report(
    validation_results=validation_results,
    output_path="./reports/validation_report.html",
    report_format="html",
    include_visualizations=True
)

# Create specific visualizations
reporter.create_visualization(
    validation_results=validation_results,
    visualization_type="confidence_distribution",
    output_path="./reports/confidence_distribution.html",
    title="Confidence Score Distribution"
)
```

### Configuration Options

```python
config = {
    # Report configuration
    "report_formats": ["html", "markdown", "json"],
    "report_title_template": "Benchmark Validation Report - {timestamp}",
    "max_results_per_page": 20,
    
    # Output configuration
    "output_directory": "./reports",
    "css_style_path": None,  # Custom CSS path
    "html_template_path": None,  # Custom HTML template
    
    # Visualization configuration
    "include_visualizations": True,
    "visualization_types": ["confidence_distribution", "metric_comparison", "validation_heatmap"],
    "theme": "light"  # or "dark"
}
```

## ValidationDashboard

The `ValidationDashboard` class provides a comprehensive dashboard for visualizing and analyzing benchmark validation results, with integration capabilities for the Distributed Testing Monitoring Dashboard.

### Features

- **Interactive Dashboards**: Create dashboards with multiple validation visualizations.
- **Comparison Dashboards**: Create comparison dashboards for analyzing multiple sets of validation results.
- **Monitoring Integration**: Integration with the Distributed Testing Monitoring Dashboard.
- **Export Capabilities**: Export dashboards to multiple formats (HTML, Markdown, JSON).
- **Embedding Support**: Embed dashboards in other systems via iframes.
- **Dashboard Management**: List, update, and delete dashboards.
- **Fallback Support**: Basic HTML dashboards when advanced visualization is unavailable.

### Usage

```python
from duckdb_api.benchmark_validation.visualization import ValidationDashboard

# Create a dashboard instance
dashboard = ValidationDashboard({
    "output_directory": "./output",
    "dashboard_directory": "dashboards",
    "monitoring_integration": True
})

# Create a dashboard
dashboard_path = dashboard.create_dashboard(
    validation_results=validation_results,
    dashboard_name="my_validation_dashboard",
    dashboard_title="My Validation Dashboard",
    dashboard_description="Dashboard for validation results"
)

# Create a comparison dashboard
comparison_dashboard_path = dashboard.create_comparison_dashboard(
    validation_results_sets={
        "baseline": baseline_validation_results,
        "experiment": experiment_validation_results
    },
    dashboard_name="validation_comparison_dashboard",
    dashboard_title="Validation Comparison Dashboard"
)

# List all available dashboards
dashboards = dashboard.list_dashboards()

# Update a dashboard
updated_path = dashboard.update_dashboard(
    dashboard_name="my_validation_dashboard",
    validation_results=new_validation_results,
    dashboard_title="Updated Validation Dashboard"
)

# Export dashboard to different formats
html_path = dashboard.export_dashboard(
    dashboard_name="my_validation_dashboard",
    export_format="html"
)

# Register with monitoring dashboard
success = dashboard.register_with_monitoring_dashboard(
    dashboard_name="my_validation_dashboard",
    page="validation",
    position="below"
)

# Get HTML for embedding dashboard in another page
iframe_html = dashboard.get_dashboard_iframe_html(
    dashboard_name="my_validation_dashboard",
    width="100%",
    height="800px"
)

# Delete a dashboard
dashboard.delete_dashboard("my_validation_dashboard")
```

### Configuration Options

```python
config = {
    # Dashboard configuration
    "dashboard_name": "benchmark_validation_dashboard",
    "dashboard_title": "Benchmark Validation Dashboard",
    "dashboard_description": "Comprehensive visualization of benchmark validation results",
    
    # Output configuration
    "output_directory": "output",
    "dashboard_directory": "dashboards",
    
    # Integration configuration
    "monitoring_integration": True,  # Enable integration with Monitoring Dashboard
    
    # Display configuration
    "theme": "light",  # or "dark"
    "auto_refresh": True,
    "refresh_interval": 300,  # 5 minutes
    
    # Content configuration
    "max_results": 1000,
    "default_view": "summary"  # or "detailed", "comparison"
}
```

### Dashboard Components

The dashboards created by the `ValidationDashboard` include the following components:

1. **Summary Metrics**: Key metrics about validation results.
2. **Confidence Distribution**: Distribution of confidence scores by validation status.
3. **Status by Benchmark Type**: Validation status distribution by benchmark type.
4. **Validation Heatmap**: Heatmap showing validation results by model and hardware.
5. **Time Series**: Validation trends over time.
6. **Results Table**: Detailed table of validation results.
7. **Issues Breakdown**: Breakdown of common validation issues.
8. **Recommendations**: Recommendations based on validation results.

## Integration with Monitoring Dashboard

The `ValidationDashboard` integrates with the Distributed Testing Monitoring Dashboard to provide a unified visualization experience:

1. Dashboards can be registered with the monitoring system.
2. Dashboards can be embedded in monitoring dashboard pages.
3. Dashboards can be accessed through the monitoring dashboard interface.

To enable this integration, set `monitoring_integration=True` in the dashboard configuration.

## Examples

For comprehensive examples, see:

- `duckdb_api.benchmark_validation.examples.reporter_example`: Example for using the ValidationReporter.
- `duckdb_api.benchmark_validation.examples.dashboard_example`: Example for using the ValidationDashboard.

## Extending the Visualization Components

The visualization components can be extended in the following ways:

1. **Custom Visualization Types**: Implement new visualization types in the reporter.
2. **Custom Dashboard Components**: Implement custom components for the dashboard.
3. **Custom Themes**: Implement custom themes for reports and dashboards.
4. **Custom Templates**: Use custom templates for HTML reports.

## Dependencies

The visualization components depend on the following packages:

- Mandatory dependencies:
  - None (basic functionality works without extra dependencies)

- Optional dependencies for enhanced functionality:
  - `pandas`: For data manipulation
  - `plotly`: For interactive visualizations
  - `matplotlib`: For static visualizations

- Optional dependencies for advanced functionality:
  - `duckdb_api.visualization.advanced_visualization`: For advanced visualization integration
  - `duckdb_api.distributed_testing.dashboard`: For monitoring dashboard integration

## Testing

Unit tests for the visualization components are available in:
- `duckdb_api.benchmark_validation.tests.test_validation_reporter`
- `duckdb_api.benchmark_validation.tests.test_validation_dashboard`

Run the tests with:
```bash
python -m duckdb_api.benchmark_validation.tests.test_validation_reporter
python -m duckdb_api.benchmark_validation.tests.test_validation_dashboard
```