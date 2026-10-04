# Simulation Database Visualization End-to-End Testing Guide

## Overview

This guide provides comprehensive information about the end-to-end testing system for the Simulation Accuracy and Validation Framework's database and visualization integration. The testing system ensures that all components work together correctly from database operations to visualization generation.

## Architecture

The end-to-end testing system is composed of three main components:

1. **Database Integration Tests**: Tests for the `SimulationValidationDBIntegration` class that validate database operations, schema management, data storage, and retrieval.

2. **Visualization Connector Tests**: Tests for the `ValidationVisualizerDBConnector` class that validate the connection between database queries and visualization generation.

3. **End-to-End Integration Tests**: Tests that validate the entire flow from database operations to visualization generation, ensuring all components work together correctly.

## Test Components

### 1. Database Integration Tests

Located in `duckdb_api/simulation_validation/test_db_integration.py`, these tests validate:

- Database initialization and schema creation
- Storage methods for simulation results, hardware results, validation results, calibration parameters, and drift detection results
- Retrieval methods for various criteria (hardware type, model type, time range, etc.)
- Analysis methods for calibration effectiveness, MAPE by hardware/model, and drift detection
- Data export methods for visualization
- Error handling and edge cases

### 2. Visualization Connector Tests

Located in `duckdb_api/simulation_validation/test_visualization_db_connector.py`, these tests validate:

- Initialization of connector with database integration and visualizer
- Methods for creating various visualizations from database queries
- Data conversion between database formats and visualization formats
- Handling of edge cases like empty datasets
- Integration with both static and interactive visualization options

### 3. End-to-End Integration Tests

Located in `duckdb_api/simulation_validation/test_e2e_visualization_db_integration.py`, these tests validate:

- The entire pipeline from database operations to visualization generation
- Real-world scenarios with actual database instances
- All visualization types and options
- Edge cases and error conditions
- Performance and reliability of the integrated system

## Running the Tests

The test runner script `run_e2e_tests.py` provides a unified interface for running all tests:

```bash
# Run all tests
python duckdb_api/simulation_validation/run_e2e_tests.py

# Run specific test suites
python duckdb_api/simulation_validation/run_e2e_tests.py --run-db       # Database integration tests only
python duckdb_api/simulation_validation/run_e2e_tests.py --run-connector # Visualization connector tests only
python duckdb_api/simulation_validation/run_e2e_tests.py --run-e2e      # End-to-end integration tests only

# Skip long-running tests
python duckdb_api/simulation_validation/run_e2e_tests.py --skip-long-tests

# Generate HTML report
python duckdb_api/simulation_validation/run_e2e_tests.py --html-report

# Generate example visualizations
python duckdb_api/simulation_validation/run_e2e_tests.py --generate-examples

# Specify output directory
python duckdb_api/simulation_validation/run_e2e_tests.py --output-dir ./test_results

# Verbose output
python duckdb_api/simulation_validation/run_e2e_tests.py --verbose
```

## Test Reports

The test runner generates several types of reports:

1. **Text Summary**: A terminal-friendly summary of test results with success/failure counts and execution times.

2. **JSON Report**: A detailed JSON report of test results that can be used for further analysis or integration with other tools.

3. **HTML Report**: A user-friendly HTML report with detailed test results, including success/failure status, execution times, and error messages.

### Example JSON Report

```json
{
  "timestamp": "2025-03-14T15:30:45.123456",
  "tests_run": 32,
  "tests_passed": 32,
  "tests_failed": 0,
  "tests_errors": 0,
  "total_time": 12.37,
  "results": [
    {
      "test_name": "duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_mape_comparison_chart",
      "status": "PASS",
      "execution_time": 0.35
    },
    {
      "test_name": "duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_hardware_comparison_heatmap",
      "status": "PASS",
      "execution_time": 0.41
    }
    // ... more test results ...
  ]
}
```

## Example Visualization Generation

The test system can generate example visualizations for documentation and demonstration purposes:

```bash
python duckdb_api/simulation_validation/run_e2e_tests.py --generate-examples --output-dir ./examples
```

This will create the following example visualizations:

- **MAPE Comparison Chart**: Comparison of Mean Absolute Percentage Error (MAPE) across different hardware types and models.
- **Hardware Comparison Heatmap**: Heatmap visualization showing performance metrics across different hardware types.
- **Time Series Chart**: Time series chart showing how metrics change over time for a specific hardware and model.
- **Drift Visualization**: Visualization of drift detection results, showing whether simulation accuracy has changed over time.
- **Calibration Improvement Chart**: Chart showing the improvement in simulation accuracy after calibration.
- **Simulation vs Hardware Chart**: Scatter plot comparing simulation predictions with actual hardware measurements.
- **Comprehensive Dashboard**: Complete dashboard with multiple visualizations for a comprehensive view of simulation accuracy.

An `index.html` file is also generated that links to all examples with descriptions.

## Adding New Tests

To add new tests to the system:

1. **Database Integration Tests**: Add new test methods to the `TestSimulationValidationDBIntegration` class in `test_db_integration.py`.

2. **Visualization Connector Tests**: Add new test methods to the `TestValidationVisualizerDBConnector` class in `test_visualization_db_connector.py`.

3. **End-to-End Tests**: Add new test methods to the `TestE2EVisualizationDBIntegration` class in `test_e2e_visualization_db_integration.py`.

All tests should follow these guidelines:

- Use descriptive method names that clearly indicate what's being tested.
- Include assertions that validate the expected behavior.
- Test both normal operation and edge cases.
- Include cleanup code to ensure tests don't interfere with each other.

## Best Practices

When working with the end-to-end testing system:

1. **Run All Tests Before Commits**: Always run the full test suite before committing changes to ensure nothing breaks.

2. **Start with Unit Tests**: When debugging, start with unit tests (database integration or visualization connector) before running end-to-end tests to isolate issues more quickly.

3. **Use Temporary Directories**: The testing system uses temporary directories by default. Don't specify an output directory unless you need to keep the results.

4. **Check HTML Reports**: Use the `--html-report` option to generate detailed HTML reports that are easier to analyze for complex failures.

5. **Generate Examples**: Use the `--generate-examples` option to create examples for documentation and to verify that visualizations are generated correctly.

6. **Skip Long Tests**: Use the `--skip-long-tests` option during development to skip time-consuming tests that are unlikely to be affected by your changes.

## Troubleshooting

Common issues and solutions:

1. **Database Connection Failures**: Ensure DuckDB is installed and the database path is correct. The test system should create the database automatically if it doesn't exist.

2. **Visualization Errors**: Most visualization errors are due to missing dependencies. Ensure Matplotlib, Plotly, and Pandas are installed.

3. **HTML Report Failures**: If you get errors generating HTML reports, install the `html-testRunner` package with `pip install html-testRunner`.

4. **Path Issues**: The test system uses absolute paths. If you see path-related errors, check that you're running the tests from the correct directory.

5. **Database Locked Errors**: If you see "database is locked" errors, ensure you're not running multiple tests that use the same database path simultaneously.

## CI/CD Integration

To integrate the end-to-end tests with CI/CD pipelines:

1. **GitHub Actions**:

```yaml
name: End-to-End Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install html-testRunner
    - name: Run tests
      run: |
        python duckdb_api/simulation_validation/run_e2e_tests.py --html-report
    - name: Archive test results
      uses: actions/upload-artifact@v2
      with:
        name: test-report
        path: temp-*/test_report.html
```

2. **GitLab CI**:

```yaml
end-to-end-tests:
  stage: test
  script:
    - pip install -r requirements.txt
    - pip install html-testRunner
    - python duckdb_api/simulation_validation/run_e2e_tests.py --html-report --output-dir ./test-output
  artifacts:
    paths:
      - ./test-output/test_report.html
```

## Conclusion

The end-to-end testing system provides comprehensive validation of the Simulation Accuracy and Validation Framework's database and visualization integration. By running these tests regularly, you can ensure that all components work together correctly and identify issues early in the development process. The system's ability to generate examples and detailed reports also makes it a valuable tool for documentation and demonstration purposes.