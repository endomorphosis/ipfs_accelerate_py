# Enhanced Visualization UI Tests

This directory contains integration tests and end-to-end test runners for the Enhanced Visualization Dashboard UI components.

## Test Files

### 1. test_enhanced_visualization_ui.py

This integration test verifies the enhanced UI components added to the visualization dashboard:

- Visualization options panel with controls for confidence intervals, trend lines, and annotations
- Export functionality with support for multiple formats (HTML, PNG, SVG, JSON, PDF)
- Callbacks for handling visualization options
- Theme integration between dashboard and visualizations

To run the tests:

```bash
python -m pytest test_enhanced_visualization_ui.py -v
```

### 2. run_enhanced_visualization_ui_e2e_test.py

This end-to-end test runner launches a live dashboard with enhanced UI features for manual testing:

1. Sets up a temporary test database with performance data containing known regressions
2. Launches the dashboard with enhanced visualization enabled
3. Provides instructions for testing the visualization options panel and export functionality
4. Allows interactive testing of theme integration and visualization options

To run the end-to-end test:

```bash
python run_enhanced_visualization_ui_e2e_test.py [--port PORT] [--no-browser] [--debug]
```

Optional arguments:
- `--port PORT`: Port to run the dashboard on (default: 8083)
- `--no-browser`: Don't open browser automatically
- `--debug`: Enable debug mode
- `--output-dir DIR`: Output directory for visualizations (default: temporary directory)
- `--db-path PATH`: Path to DuckDB database (default: temporary file)

## Visualization Options

The enhanced UI adds the following visualization options:

1. **Show Confidence Intervals**: Displays statistical confidence intervals around the data before and after detected change points
2. **Show Trend Lines**: Adds trend lines to visualize the slope of the data before and after detected change points
3. **Show Annotations**: Adds annotations with percentage change and statistical significance to the visualization

## Export Formats

The enhanced UI supports exporting visualizations in the following formats:

- HTML: Interactive web page with full interactivity
- PNG: Static image format
- SVG: Scalable vector graphics format
- JSON: Raw JSON representation of the visualization
- PDF: Portable document format

## Integration with Distributed Testing Framework

These tests verify that the enhanced visualization UI integrates with the Distributed Testing Framework by:

1. Using the same data sources and APIs as the rest of the framework
2. Sharing configuration and state with the framework
3. Following the same design patterns and UI conventions
4. Supporting the same themes and styling

## Running in CI/CD

These tests can be run as part of continuous integration by adding them to the test suite:

```bash
# Add these tests to your CI pipeline
python -m pytest duckdb_api/distributed_testing/tests/test_enhanced_visualization_ui.py
```

The end-to-end test can be run in headless mode with the `--no-browser` flag for manual inspection environments.