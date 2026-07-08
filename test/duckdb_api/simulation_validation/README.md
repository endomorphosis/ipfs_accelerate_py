# Simulation Accuracy and Validation Framework

This directory contains the Simulation Accuracy and Validation Framework, which provides a comprehensive system for validating, calibrating, and monitoring hardware simulation accuracy. The framework ensures that simulation results closely match real hardware performance, enabling reliable performance predictions for untested hardware configurations.

The framework includes a web-based user interface for interactive management of validation tasks, reporting, and monitoring, with full authentication, notification, and job management capabilities.

## Key Components

The framework consists of several key components:

- `simulation_validation_framework.py`: Main integration module that orchestrates all components
- `core/`: Core interfaces and base classes for the framework
- `comparison/`: Components for comparing simulation and hardware results
- `statistical/`: Statistical validation tools for analyzing simulation accuracy
- `calibration/`: Calibration system for improving simulation accuracy
- `drift_detection/`: Drift detection components for monitoring changes in accuracy
- `visualization/`: Visualization and reporting components for presenting results
- `db_integration.py`: Database integration for persistent storage of validation data
- `visualization/validation_visualizer_db_connector.py`: Connector between database and visualization systems with monitoring dashboard integration
- `ui/`: Web user interface for the framework with authentication, job management and notification systems
- `database_predictive_analytics.py`: Time series forecasting and predictive analytics for database performance metrics with anomaly detection, threshold-based alerts, proactive recommendations, visualization generation, and ensemble forecasting for improved accuracy. Supports multiple visualization formats with robust error handling.

## Component Documentation

Each major component has its own dedicated documentation:

- [Visualization Components](visualization/README.md): Documentation for visualization components
- [Calibration Components](calibration/README.md): Documentation for calibration system
- [Drift Detection Components](drift_detection/README.md): Documentation for drift detection system
- [Monitoring Dashboard Integration](MONITORING_DASHBOARD_INTEGRATION.md): Documentation for real-time monitoring dashboard integration
- [CI/CD Integration](CI_CD_INTEGRATION.md): Documentation for CI/CD workflow and automation
- [Web UI Components](ui/README.md): Documentation for web user interface components
- [Database Predictive Analytics](API_DOCUMENTATION.md): Documentation for database performance prediction and proactive optimization

## End-to-End Testing System

The framework includes a comprehensive end-to-end testing system that validates the entire flow from database operations to visualization generation. This ensures that all components work together correctly in real-world scenarios.

### Testing Components

- `test_e2e_visualization_db_integration.py`: End-to-end tests for visualization database integration
- `test_visualization_db_connector.py`: Tests for the database connector component
- `run_e2e_tests.py`: Test runner script with summary reporting and example generation
- `run_visualization_tests.sh`: Shell script wrapper for running visualization tests

### Running Tests

Use the provided shell scripts to run tests:

```bash
# Run all tests with the general runner
./run_simulation_validation_tests.sh

# Options for the general test runner
./run_simulation_validation_tests.sh --html-report --output-dir ./reports
./run_simulation_validation_tests.sh --generate-examples
./run_simulation_validation_tests.sh --run-e2e

# Run visualization tests specifically
./run_visualization_tests.sh

# Options for the visualization test runner
./run_visualization_tests.sh --test-type mape
./run_visualization_tests.sh --generate-examples --output-dir ./examples
./run_visualization_tests.sh --test-type drift --interactive
```

### Web User Interface

The framework includes a comprehensive web-based user interface for managing validation tasks, viewing reports, and monitoring simulation accuracy. The Web UI includes:

- **User Authentication**: Secure login system with role-based access control
- **Job Management**: Schedule and monitor long-running validation, calibration, and reporting tasks
- **Notification System**: Real-time notifications for important system events
- **User Preferences**: Customizable interface settings for each user
- **Reporting Interface**: Interactive report generation and visualization
- **Dashboard Access**: Access to comprehensive monitoring dashboards
- **CI/CD Integration**: Integration with CI/CD systems via API and webhooks

### Running the Web UI

```bash
# Start the web UI on localhost port 5000
python -m duckdb_api.simulation_validation.ui.app --host localhost --port 5000

# Start with a custom configuration file
python -m duckdb_api.simulation_validation.ui.app --config config.json

# Start in debug mode
python -m duckdb_api.simulation_validation.ui.app --debug
```

### Web UI Screenshots

![Dashboard Screenshot](ui/screenshots/dashboard.png)
![Validation Results Screenshot](ui/screenshots/validation_results.png)
![Job Management Screenshot](ui/screenshots/jobs.png)

## CI/CD Integration

The framework includes CI/CD integration through GitHub Actions. The workflow automatically validates simulation results, analyzes them for issues, and publishes interactive dashboards.

```bash
# Run the test coverage analyzer 
python -m duckdb_api.simulation_validation.analyze_test_coverage \
  --coverage-file test_results/coverage.xml \
  --output-format html \
  --output-file coverage_report.html

# Analyze validation results
python -m duckdb_api.simulation_validation.analyze_validation_results \
  --results-dir validation_output \
  --output-format markdown \
  --output-file validation_analysis.md

# Detect validation issues
python -m duckdb_api.simulation_validation.detect_validation_issues \
  --results-dir validation_output \
  --threshold 0.1 \
  --output-format html \
  --output-file validation_issues.html

# Generate visualization dashboard
python -m duckdb_api.simulation_validation.visualization.generate_dashboard \
  --input-dir validation_output \
  --output-dir dashboard \
  --interactive \
  --title "Simulation Validation Dashboard"
```

For detailed documentation on the CI/CD integration, see [CI/CD Integration](CI_CD_INTEGRATION.md).

## Usage Examples

### Basic Framework Usage

```python
from duckdb_api.simulation_validation import get_framework_instance
from duckdb_api.simulation_validation import SimulationResult, HardwareResult

# Initialize framework
framework = get_framework_instance()

# Generate or load simulation and hardware results
simulation_results = [...]  # List of SimulationResult objects
hardware_results = [...]    # List of HardwareResult objects

# Run validation
validation_results = framework.validate(
    simulation_results=simulation_results,
    hardware_results=hardware_results,
    protocol="standard"
)

# Generate a summary
summary = framework.summarize_validation(validation_results)
print(f"Overall MAPE: {summary['overall']['mape']['mean']:.2f}%")
print(f"Status: {summary['overall']['status']}")

# Generate a report
report = framework.generate_report(
    validation_results=validation_results,
    format="markdown"
)
```

### Database Integration

```python
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration

# Initialize database integration
db_integration = SimulationValidationDBIntegration(
    db_path="./benchmark_db.duckdb"
)

# Initialize database schema
db_integration.initialize_database()

# Store validation results in database
db_integration.store_validation_results(validation_results)

# Get validation results by criteria
hw_model_results = db_integration.get_validation_results_by_criteria(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased"
)
```

### Visualization with Database Connector

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

# Create a comprehensive dashboard
connector.create_comprehensive_dashboard_from_db(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    output_path="./dashboard.html"
)
```

### Drift Detection

```python
from duckdb_api.simulation_validation.drift_detection.advanced_detector import AdvancedDriftDetector

# Create the detector
detector = AdvancedDriftDetector()

# Detect drift using database
drift_result = detector.detect_drift_from_db(
    db_integration=db_integration,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    historical_window_start="2025-01-01",
    historical_window_end="2025-01-31",
    new_window_start="2025-02-01",
    new_window_end="2025-02-28"
)

# Check if significant drift was detected
if drift_result.is_significant:
    print(f"Significant drift detected in {drift_result.hardware_type} - {drift_result.model_type}")
    
    # Analyze drift by metrics
    for metric, details in drift_result.drift_metrics.items():
        if details["drift_detected"]:
            print(f"  Drift in {metric}: {details['mean_change_pct']:.2f}% change (p-value: {details['p_value']:.4f})")
    
    # Visualize the drift detection results
    connector.create_drift_visualization_from_db(
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        output_path="drift_visualization.html"
    )
```

### Calibration

```python
from duckdb_api.simulation_validation.calibration.advanced_calibrator import AdvancedCalibrator

# Create the calibrator
calibrator = AdvancedCalibrator()

# Calibrate using database results
calibration_record = calibrator.calibrate_from_db(
    db_integration=db_integration,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased"
)

# Store calibration record in database
db_integration.store_calibration_record(calibration_record)

# Visualize calibration effectiveness
connector.create_calibration_improvement_chart_from_db(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    output_path="./calibration_improvement.html"
)
```

### Database Predictive Analytics

The framework includes a comprehensive Database Predictive Analytics component (`database_predictive_analytics.py`) that provides time series forecasting for database performance metrics, proactive issue detection, and optimization recommendations:

```python
from duckdb_api.simulation_validation.database_predictive_analytics import DatabasePredictiveAnalytics
from duckdb_api.simulation_validation.automated_optimization_manager import get_optimization_manager
from duckdb_api.simulation_validation.db_performance_optimizer import get_db_optimizer

# Create database optimizer and automated optimization manager
db_optimizer = get_db_optimizer(db_path="./benchmark_db.duckdb")
auto_manager = get_optimization_manager(db_optimizer=db_optimizer)

# Create predictive analytics instance
predictive = DatabasePredictiveAnalytics(
    automated_optimization_manager=auto_manager
)

# Generate forecasts
forecast_result = predictive.forecast_database_metrics(
    horizon="medium_term",
    specific_metrics=["query_time", "storage_size"]
)

# Generate visualizations
vis_result = predictive.generate_forecast_visualizations(
    forecast_results=forecast_result,
    output_format="file"
)

# Check for predicted threshold alerts
alert_result = predictive.check_predicted_thresholds(
    forecast_results=forecast_result
)

# Get proactive recommendations
rec_result = predictive.recommend_proactive_actions(
    forecast_results=forecast_result,
    threshold_alerts=alert_result
)

# Run comprehensive analysis
analysis_result = predictive.analyze_database_health_forecast(
    horizon="medium_term",
    generate_visualizations=True,
    output_format="file"
)
```

The component implements multiple forecasting methods:
- **ARIMA**: For time series with temporal dependencies
- **Exponential Smoothing**: For time series with level, trend, and seasonal components
- **Linear Regression**: For time series with linear trends
- **Ensemble Forecasting**: Combines all methods for improved accuracy

Key features include:
- Trend analysis with direction and magnitude detection
- Anomaly detection in both historical and forecasted data
- Confidence intervals for predictions
- Threshold-based alerting with configurable sensitivity
- Proactive recommendation system with urgency classification
- Comprehensive visualization with customizable themes
- Support for multiple output formats (base64, file, object)
- Graceful degradation when optional dependencies are unavailable

The Database Predictive Analytics component is integrated with the CLI via the `predictive` command in `run_database_performance_monitoring.py`:

```bash
# Generate a forecast for database metrics
python run_database_performance_monitoring.py predictive --action forecast --horizon medium_term

# Generate visualizations for forecasted metrics
python run_database_performance_monitoring.py predictive --action visualize --format html --output forecast_viz.html

# Check for predicted threshold alerts
python run_database_performance_monitoring.py predictive --action alerts

# Get proactive recommendations
python run_database_performance_monitoring.py predictive --action recommend

# Run comprehensive analysis
python run_database_performance_monitoring.py predictive --action analyze --horizon medium_term --visualize

# Generate a dark-themed visualization
python run_database_performance_monitoring.py predictive --action visualize --theme dark
```

For detailed documentation of the Database Predictive Analytics API, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Implementation Status

The framework implementation status is tracked in the [SIMULATION_ACCURACY_VALIDATION_IMPLEMENTATION.md](../../SIMULATION_ACCURACY_VALIDATION_IMPLEMENTATION.md) file in the project root directory. The current status shows:

- ✅ Core components and interface definition
- ✅ Validation methodology implementation
- ✅ Comparison pipeline implementation
- ✅ Basic and advanced statistical validation
- ✅ Calibration system implementation
- ✅ Drift detection implementation
- ✅ Visualization and reporting implementation
- ✅ Database integration implementation
- ✅ End-to-end testing system implementation
- ✅ Statistical validation enhancements
- ✅ Calibration system implementation
- ✅ Monitoring dashboard integration
- ✅ Database Predictive Analytics system (COMPLETED - July 15, 2025)
- ✅ CI/CD integration
- ✅ User authentication and authorization
- ✅ Job scheduling for long-running operations
- ✅ Notification system for important events
- ✅ User preferences and settings management
- 🔄 Comprehensive reporting system (IN PROGRESS)

## Dependencies

The Simulation Accuracy and Validation Framework has the following dependencies:

- **Core Dependencies**:
  - numpy: Numerical operations
  - pandas: Data manipulation and analysis
  - duckdb: Database storage and querying

- **Visualization Dependencies**:
  - matplotlib: Static visualization
  - plotly: Interactive visualization

- **Statistical Dependencies**:
  - scipy: Scientific computing and statistical tests
  - scikit-learn: Machine learning algorithms
  - statsmodels: Statistical models

## Contributing

When contributing to the Simulation Accuracy and Validation Framework, please follow these guidelines:

1. **Follow the Existing Architecture**: Respect the modular design of the framework
2. **Add Tests**: Include unit tests for new functionality and update end-to-end tests as needed
3. **Update Documentation**: Update the relevant README files and the implementation status document
4. **Follow Coding Standards**: Maintain consistent code style and documentation

## License

The Simulation Accuracy and Validation Framework is part of the IPFS Accelerate Python project and follows the same licensing terms.