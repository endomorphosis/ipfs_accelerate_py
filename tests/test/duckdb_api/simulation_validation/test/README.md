# Simulation Validation Framework Test Suite

This directory contains comprehensive tests for the Simulation Accuracy and Validation Framework, including unit tests, integration tests, and end-to-end tests. The test suite is designed to validate all aspects of the framework, from database integration to visualization generation to end-to-end workflows.

## Test Components

The test suite consists of the following components:

1. **Test Data Generator** (`test_data_generator.py`)
   - Generates realistic test data for simulation results, hardware results, validation results, calibration records, and drift detection
   - Creates various scenarios including calibration and drift
   - Supports time series generation with trends, seasonality, and outliers

2. **Comprehensive E2E Tests** (`test_comprehensive_e2e.py`)
   - Tests all aspects of the framework in a comprehensive and realistic way
   - Validates database operations, visualization generation, and end-to-end workflows
   - Measures performance with larger datasets
   - Generates performance reports

3. **Visualization Tests** (`test_visualization.py`)
   - Tests the visualization components of the framework
   - Validates various visualization types (metric comparisons, heatmaps, error distributions, etc.)
   - Tests interactive and static visualizations

4. **Database Integration Tests** (`test_db_integration.py`)
   - Tests the database integration components
   - Validates storage, retrieval, and querying of all data types

5. **DB-Visualization Connector Tests** (`test_visualization_db_connector.py`)
   - Tests the connection between database and visualization components
   - Validates chart generation from database data

## How to Run the Tests

You can run the tests using the `run_e2e_tests.py` script in the parent directory. Here are some common usage examples:

### Run All Tests

```bash
python run_e2e_tests.py
```

### Run Specific Test Types

```bash
# Run database integration tests only
python run_e2e_tests.py --run-db

# Run visualization connector tests only
python run_e2e_tests.py --run-connector

# Run standard end-to-end tests only
python run_e2e_tests.py --run-e2e

# Run comprehensive end-to-end tests only
python run_e2e_tests.py --run-comprehensive
```

### Skip Long-Running Tests

```bash
python run_e2e_tests.py --skip-long-tests
```

### Generate Reports

```bash
# Generate HTML report
python run_e2e_tests.py --html-report

# Generate performance report
python run_e2e_tests.py --performance-report

# Specify output directory
python run_e2e_tests.py --output-dir /path/to/output
```

### Generate Example Visualizations

```bash
python run_e2e_tests.py --generate-examples
```

### Verbose Output

```bash
python run_e2e_tests.py --verbose
```

## Running Individual Test Files

You can also run individual test files directly using Python's unittest framework:

```bash
# Run the comprehensive E2E tests
python -m unittest test.test_comprehensive_e2e

# Run the test data generator tests
python -m unittest test_data_generator.py

# Run a specific test class
python -m unittest test.test_comprehensive_e2e.TestComprehensiveEndToEnd

# Run a specific test method
python -m unittest test.test_comprehensive_e2e.TestComprehensiveEndToEnd.test_01_database_initialization
```

## Test Data Generation

The test data generator can be used independently to generate test data for various scenarios:

```python
from test.test_data_generator import TestDataGenerator

# Create a data generator with a fixed seed for reproducibility
generator = TestDataGenerator(seed=42)

# Generate a complete dataset with multiple models and hardware types
dataset = generator.generate_complete_dataset(
    num_models=3,
    num_hardware_types=3,
    days_per_series=30,
    include_calibrations=True,
    include_drifts=True
)

# Generate a specific scenario - calibration
hw_results, sim_results, val_results, calibration_record = generator.generate_calibration_scenario(
    model_id="bert-base-uncased",
    hardware_id="gpu_rtx3080"
)

# Generate a specific scenario - drift
hw_results, sim_results, val_results, drift_record = generator.generate_drift_scenario(
    model_id="vit-base-patch16-224",
    hardware_id="cpu_intel_xeon"
)

# Save dataset to JSON file
generator.save_dataset_to_json(dataset, "test_dataset.json")
```

## Understanding Test Results

The test runner will generate a summary of test results, including:

- Number of tests run
- Number of tests passed, failed, and with errors
- Total execution time
- Detailed results for each test

If you use the `--html-report` option, a comprehensive HTML report will be generated with more detailed information.

If you use the `--performance-report` option, detailed performance metrics will be generated in JSON and Markdown formats.

## Adding New Tests

When adding new tests:

1. Follow the existing naming conventions
2. Use descriptive names that indicate what is being tested
3. Include docstrings that explain the test purpose
4. Group related tests in the same test class
5. Maintain test independence (tests should not depend on the results of other tests)

## Notes for Contributors

- The test data generator is designed to be extensible. If you need to generate new types of test data, add methods to the TestDataGenerator class.
- The comprehensive E2E tests are numbered to ensure they run in a specific order. When adding new tests, follow the existing numbering pattern.
- Performance tests are marked with "performance" in their names so they can be easily identified and skipped when needed.
- The test suite is designed to be run in CI/CD environments, so all tests should be fully automated and not require user interaction.
- Tests should clean up after themselves to avoid interference with other tests.