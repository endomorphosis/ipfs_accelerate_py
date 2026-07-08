# API Metrics Validation Guide

This guide provides detailed documentation for the API Metrics Validation module in the IPFS Accelerate Python Framework. The API Metrics Validation module is part of the broader Simulation Validation system and provides comprehensive tools for validating API performance metrics quality, prediction accuracy, anomaly detection effectiveness, and recommendation relevance.

## Overview

The API Metrics Validation module offers the following capabilities:

1. **Database Integration**: Store and retrieve API metrics, predictions, anomalies, and recommendations in a DuckDB database
2. **Data Quality Validation**: Assess the completeness, consistency, validity, and timeliness of API metrics data
3. **Prediction Accuracy Validation**: Evaluate the accuracy of API performance predictions
4. **Anomaly Detection Validation**: Measure the effectiveness of anomaly detection
5. **Recommendation Relevance Validation**: Analyze the relevance, actionability, and impact coverage of recommendations
6. **Comprehensive Reporting**: Generate detailed validation reports with actionable recommendations

## Architecture

The module consists of the following key components:

- **DuckDBAPIMetricsRepository**: Provides data access methods for API metrics stored in DuckDB
- **APIMetricsValidator**: Implements validation logic for API metrics quality and predictions
- **run_api_metrics_validation.py**: Command-line tool for running validations

## Installation and Dependencies

The API Metrics Validation module requires the following dependencies:

- DuckDB
- NumPy
- Pandas
- scikit-learn

These dependencies should be installed as part of the standard IPFS Accelerate Python Framework installation.

## Using the API Metrics Repository

The `DuckDBAPIMetricsRepository` class provides methods for storing and retrieving API metrics data in a DuckDB database.

```python
from duckdb_api.simulation_validation.api_metrics import DuckDBAPIMetricsRepository

# Create repository instance
repository = DuckDBAPIMetricsRepository(
    db_path="api_metrics.duckdb",
    create_if_missing=True
)

# Store API metrics
metric_id = repository.store_metric({
    'timestamp': datetime.now(),
    'endpoint': '/v1/completions',
    'model': 'gpt-4',
    'response_time': 1.25,
    'status_code': 200,
    'tokens': 150,
    'success': True
})

# Retrieve API metrics
metrics = repository.get_metrics(
    start_time=datetime.now() - timedelta(days=7),
    endpoint='/v1/completions',
    model='gpt-4',
    limit=100
)

# Store a prediction
prediction_id = repository.store_prediction({
    'timestamp': datetime.now(),
    'endpoint': '/v1/completions',
    'model': 'gpt-4',
    'predicted_value': 1.35,
    'prediction_type': 'response_time',
    'confidence': 0.85,
    'horizon': 24
})

# Generate sample data (for testing)
repository.generate_sample_data(
    num_records=1000,
    days_back=30
)
```

## Using the API Metrics Validator

The `APIMetricsValidator` class provides methods for validating various aspects of API metrics data.

```python
from duckdb_api.simulation_validation.api_metrics import APIMetricsValidator

# Create validator instance
validator = APIMetricsValidator(repository=repository)

# Validate data quality
quality_results = validator.validate_data_quality(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    endpoint='/v1/completions',
    model='gpt-4'
)

# Validate prediction accuracy
prediction_results = validator.validate_prediction_accuracy(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    endpoint='/v1/completions',
    model='gpt-4',
    prediction_type='response_time'
)

# Generate comprehensive validation report
report = validator.generate_validation_report(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

# Access validation results
overall_score = report['overall_score']
data_quality = report['data_quality']['overall_quality']
prediction_accuracy = report['prediction_accuracy']['accuracy']
```

## Command-Line Interface

The API Metrics Validation module includes a command-line interface for running validations.

```bash
# Generate sample data and run full validation
python test/run_api_metrics_validation.py --db-path=api_metrics.duckdb --generate-sample --output=validation_report.json

# Run specific validation type
python test/run_api_metrics_validation.py --db-path=api_metrics.duckdb --report-type=data-quality

# Filter by endpoint or model
python test/run_api_metrics_validation.py --db-path=api_metrics.duckdb --endpoint=/v1/completions --model=gpt-4
```

### Command-Line Options

The `run_api_metrics_validation.py` script supports the following options:

- `--db-path`: Path to DuckDB database file (default: api_metrics.duckdb)
- `--generate-sample`: Generate sample data in the DuckDB database
- `--num-samples`: Number of sample records to generate (default: 1000)
- `--days-back`: Number of days back to generate data for (default: 30)
- `--endpoint`: Filter by specific API endpoint
- `--model`: Filter by specific model
- `--output`: Path to output JSON file for validation results
- `--report-type`: Type of validation report to generate (full, data-quality, prediction, anomaly, recommendation)

## Integration with API Management UI

The API Metrics Validation module integrates with the API Management UI to provide real-time validation of API metrics and recommendations.

To enable DuckDB integration in the API Management UI:

```bash
python test/run_api_management_ui.py --db-path=api_metrics.duckdb
```

The API Management UI with DuckDB integration provides:

1. Enhanced data storage and retrieval performance
2. Persistent storage of API metrics, predictions, anomalies, and recommendations
3. Real-time validation of data quality and prediction accuracy
4. Advanced filtering and querying capabilities
5. Comprehensive visualization of validation results

## Validation Metrics and Thresholds

The API Metrics Validator uses the following validation metrics and thresholds:

### Data Quality Validation

- **Completeness**: Percentage of required fields present
- **Consistency**: Percentage of values within expected ranges
- **Validity**: Percentage of logical relationships maintained
- **Timeliness**: Percentage of timestamps that are recent and reasonable
- **Threshold**: 95% completeness required

### Prediction Accuracy Validation

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of Determination
- **Accuracy**: 1 - normalized MAE
- **Threshold**: 80% accuracy required

### Anomaly Detection Validation

- **Precision**: Percentage of detected anomalies that are actual anomalies
- **Recall**: Percentage of actual anomalies that are detected
- **F1 Score**: Harmonic mean of precision and recall
- **Threshold**: 70% precision required

### Recommendation Relevance Validation

- **Relevance Score**: Relevance of recommendations to actual issues
- **Actionability Score**: Ease of implementing recommendations
- **Impact Coverage**: Percentage of issues addressed by recommendations
- **Threshold**: 60% overall quality required

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Ensure the DuckDB file path is valid and accessible
   - Check if the directory has write permissions

2. **Import Errors**:
   - Ensure all dependencies are installed
   - Check the Python path includes the project root directory

3. **Validation Errors**:
   - Ensure the database contains sufficient data for validation
   - Check the date ranges for validation are appropriate

### Logging

The API Metrics Validation module uses the standard Python logging system. To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Data Collection**:
   - Collect API metrics consistently across all endpoints
   - Include all required fields in metrics
   - Store timestamps in ISO format or as datetime objects

2. **Validation Configuration**:
   - Adjust validation thresholds based on your specific requirements
   - Run validations regularly to monitor data quality
   - Use appropriate date ranges for validation

3. **Performance Optimization**:
   - Use batch operations for storing multiple metrics
   - Create indices for frequently queried fields
   - Close the database connection when finished

## Example Workflows

### Basic Validation Workflow

```python
from duckdb_api.simulation_validation.api_metrics import DuckDBAPIMetricsRepository, APIMetricsValidator
from datetime import datetime, timedelta

# Create repository instance
repository = DuckDBAPIMetricsRepository(db_path="api_metrics.duckdb")

# Create validator instance
validator = APIMetricsValidator(repository=repository)

# Set time range for validation
end_time = datetime.now()
start_time = end_time - timedelta(days=30)

# Validate data quality
quality_results = validator.validate_data_quality(
    start_time=start_time,
    end_time=end_time
)

# Print validation results
print(f"Data Quality Score: {quality_results['overall_quality']:.2f}")
print(f"Threshold Met: {quality_results['threshold_met']}")

# Print recommendations
for i, rec in enumerate(quality_results['recommendations'], 1):
    print(f"{i}. [{rec['priority']}] {rec['issue']}: {rec['recommendation']}")
```

### Comprehensive Validation Workflow

```python
from duckdb_api.simulation_validation.api_metrics import DuckDBAPIMetricsRepository, APIMetricsValidator
from datetime import datetime, timedelta
import json

# Create repository instance
repository = DuckDBAPIMetricsRepository(db_path="api_metrics.duckdb")

# Create validator instance
validator = APIMetricsValidator(repository=repository)

# Generate comprehensive validation report
report = validator.generate_validation_report()

# Save report to file
with open("validation_report.json", "w") as f:
    json.dump(report, f, default=str, indent=2)

# Print overall results
print(f"Overall Score: {report['overall_score']:.2f}")
print(f"Status: {report['status']}")

# Print component scores
print("\nComponent Scores:")
print(f"Data Quality: {report['data_quality']['overall_quality']:.2f}")
print(f"Prediction Accuracy: {report['prediction_accuracy']['accuracy']:.2f}")
print(f"Anomaly Detection: {report['anomaly_detection']['effectiveness']:.2f}")
print(f"Recommendation Quality: {report['recommendation_relevance']['overall_quality']:.2f}")
```