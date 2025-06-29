# Visualization Components for Simulation Accuracy and Validation Framework

This directory contains the visualization components for the Simulation Accuracy and Validation Framework. These components generate visualizations for simulation validation results, calibration history, drift detection, and other metrics.

## Key Components

- `validation_visualizer.py`: Core visualization engine that creates various charts and dashboards
- `validation_visualizer_db_connector.py`: Connector between database and visualization components
- `validation_reporter.py`: Generates comprehensive validation reports with visualizations

## Visualization Types

The framework supports the following visualization types:

1. **MAPE Comparison Chart**: Compare Mean Absolute Percentage Error (MAPE) across different hardware types and models
2. **Hardware Comparison Heatmap**: Visualize performance metrics across different hardware types
3. **Time Series Chart**: Show how metrics change over time for a specific hardware and model
4. **Drift Visualization**: Visualize drift detection results, showing whether simulation accuracy has changed over time
5. **Calibration Improvement Chart**: Show the improvement in simulation accuracy after calibration
6. **Simulation vs Hardware Chart**: Compare simulation predictions with actual hardware measurements
7. **Comprehensive Dashboard**: Complete dashboard with multiple visualizations for a comprehensive view of simulation accuracy
8. **Metric Importance Chart**: Show relative importance of different metrics in validation results
9. **Calibration Effectiveness Analysis**: Analyze effectiveness of calibration across different hardware/model combinations
10. **Metrics Over Time**: Track metrics aggregated over time to identify trends and patterns

## End-to-End Testing System

The visualization components are tested through a comprehensive end-to-end testing system that validates the entire flow from database operations to visualization generation. This ensures that all components work together correctly in real-world scenarios.

### Testing Components

- `test_e2e_visualization_db_integration.py`: End-to-end tests for the visualization database integration
- `test_visualization_db_connector.py`: Tests for the database connector component
- `run_e2e_tests.py`: Test runner script with summary reporting and example generation
- `run_visualization_tests.sh`: Dedicated script for running visualization tests

### Running Tests

To run visualization tests, use the provided shell script:

```bash
# Run all visualization tests
./run_visualization_tests.sh

# Run specific test type
./run_visualization_tests.sh --test-type mape

# Generate example visualizations
./run_visualization_tests.sh --generate-examples --output-dir ./examples

# Run interactive visualizations
./run_visualization_tests.sh --test-type drift --interactive
```

Valid test types include:
- `all`: All visualization tests
- `connector`: Tests for the validation_visualizer_db_connector
- `e2e`: All end-to-end visualization tests
- `mape`: MAPE comparison chart tests
- `hardware`: Hardware comparison heatmap tests
- `time`: Time series chart tests
- `drift`: Drift visualization tests
- `calibration`: Calibration improvement chart tests
- `comprehensive`: Comprehensive dashboard tests

### Test Data Generation

The test suite automatically generates realistic test data including:
- Simulation results
- Hardware results
- Validation results
- Calibration records
- Drift detection results

This ensures comprehensive testing of all visualization components with realistic data scenarios.

### Example Generation

The test runner can generate example visualizations for documentation and demos:

```bash
./run_visualization_tests.sh --generate-examples --output-dir ./examples
```

This generates examples of all visualization types and creates an `index.html` file with links to all examples.

### HTML and JSON Reporting

The test runner can generate both HTML and JSON reports of test results:

```bash
./run_visualization_tests.sh --html-report --output-dir ./reports
```

HTML reports include detailed test results with pass/fail status and execution times, while JSON reports provide structured data for further analysis.

## Using the Visualization Components

### Basic Usage with Database Connector

```python
from duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector

# Create the connector
connector = ValidationVisualizerDBConnector(db_path="./simulation_db.duckdb")

# Create a MAPE comparison chart
connector.create_mape_comparison_chart_from_db(
    hardware_ids=["gpu_rtx3080", "cpu_intel_xeon"],
    model_ids=["bert-base-uncased"],
    metric_name="throughput_items_per_second",
    output_path="./mape_comparison.html"
)

# Create a hardware comparison heatmap
connector.create_hardware_comparison_heatmap_from_db(
    metric_name="average_latency_ms",
    model_ids=["bert-base-uncased", "vit-base-patch16-224"],
    output_path="./hardware_heatmap.html"
)

# Create a time series chart
connector.create_time_series_chart_from_db(
    metric_name="throughput_items_per_second",
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    output_path="./time_series.html"
)

# Create a comprehensive dashboard
connector.create_comprehensive_dashboard_from_db(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    output_path="./dashboard.html"
)
```

### Direct Visualization without Database

```python
from duckdb_api.simulation_validation.visualization.validation_visualizer import ValidationVisualizer
from duckdb_api.simulation_validation.core.base import SimulationResult, HardwareResult, ValidationResult

# Create the visualizer
visualizer = ValidationVisualizer()

# Create validation results
# (example code to create SimulationResult, HardwareResult, and ValidationResult objects)

# Create a MAPE comparison chart
visualizer.create_mape_comparison_chart(
    validation_results=validation_results,
    metric_name="throughput_items_per_second",
    hardware_ids=["gpu_rtx3080", "cpu_intel_xeon"],
    output_path="./mape_comparison.html"
)
```

### Creating a Visualization Report

```python
from duckdb_api.simulation_validation.visualization.validation_reporter import ValidationReporter

# Create the reporter
reporter = ValidationReporter()

# Generate a report
reporter.generate_validation_report(
    validation_results=validation_results,
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    output_path="./validation_report.html"
)
```

## Dependencies

The visualization components require the following dependencies:
- pandas: Data manipulation and analysis
- matplotlib: Static visualization
- plotly: Interactive visualization
- numpy: Numerical operations

## Integration with Monitoring Dashboard

The visualization components can be integrated with the Monitoring Dashboard for centralized visualization:

```python
from duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector
from duckdb_api.distributed_testing.run_monitoring_dashboard import dashboard_client

# Create the connector
connector = ValidationVisualizerDBConnector()

# Create a visualization
vis_html = connector.create_mape_comparison_chart_from_db(
    hardware_ids=["gpu_rtx3080", "cpu_intel_xeon"],
    model_ids=["bert-base-uncased"],
    metric_name="throughput_items_per_second",
    interactive=True
)

# Add to dashboard
dashboard_client.add_panel(
    title="MAPE Comparison",
    panel_type="html",
    content=vis_html,
    group="Simulation Validation"
)
```