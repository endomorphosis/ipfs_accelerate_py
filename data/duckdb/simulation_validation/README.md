# Simulation Validation Module

The Simulation Validation module provides tools for validating and calibrating simulation data, comparing real-world performance with simulated performance, and generating reports on simulation accuracy.

## Module Structure

- **api_metrics/**: API Metrics Validation tools
  - `api_metrics_repository.py`: DuckDB repository for API metrics
  - `api_metrics_validator.py`: Validator for API metrics quality and predictions

- **calibration/**: Calibration tools for simulation models
  - `basic_calibrator.py`: Basic calibration functionality
  - `advanced_calibrator.py`: Advanced calibration tools with uncertainty quantification
  - `cross_validation.py`: Cross-validation tools for calibration
  - `parameter_discovery.py`: Tools for discovering optimal parameters

- **reporting/**: Reporting tools for simulation validation
  - `comparative_report.py`: Generate comparative reports
  - `executive_summary.py`: Generate executive summaries
  - `technical_report.py`: Generate technical reports

## API Metrics Validation

The API Metrics Validation module provides comprehensive tools for validating the quality and effectiveness of API performance metrics, predictions, anomaly detection, and recommendations. It supports the following validation types:

1. **Data Quality Validation**: Validates the completeness, consistency, validity, and timeliness of API metrics data.

2. **Prediction Accuracy Validation**: Validates the accuracy of API performance predictions by comparing them with actual metrics.

3. **Anomaly Detection Validation**: Validates the effectiveness of anomaly detection by comparing detected anomalies with statistical anomalies.

4. **Recommendation Relevance Validation**: Validates the relevance, actionability, and impact coverage of API performance recommendations.

### Usage Example

```python
from duckdb_api.simulation_validation.api_metrics import DuckDBAPIMetricsRepository, APIMetricsValidator

# Create repository instance
repository = DuckDBAPIMetricsRepository(
    db_path="api_metrics.duckdb",
    create_if_missing=True
)

# Generate sample data (for testing)
repository.generate_sample_data(
    num_records=1000,
    days_back=30
)

# Create validator instance
validator = APIMetricsValidator(repository=repository)

# Generate comprehensive validation report
report = validator.generate_validation_report(
    start_time=None,  # Default: 30 days ago
    end_time=None,    # Default: current time
    endpoint=None,    # Optional filter by endpoint
    model=None        # Optional filter by model
)

# Access validation results
overall_score = report['overall_score']
data_quality = report['data_quality']['overall_quality']
prediction_accuracy = report['prediction_accuracy']['accuracy']
anomaly_effectiveness = report['anomaly_detection']['effectiveness']
recommendation_quality = report['recommendation_relevance']['overall_quality']

# Get recommendations
recommendations = report['all_recommendations']
```

### Command-Line Interface

The module includes a command-line interface for running validations:

```bash
# Generate sample data and run full validation
python test/run_api_metrics_validation.py --db-path=api_metrics.duckdb --generate-sample --output=validation_report.json

# Run specific validation type
python test/run_api_metrics_validation.py --db-path=api_metrics.duckdb --report-type=data-quality

# Filter by endpoint or model
python test/run_api_metrics_validation.py --db-path=api_metrics.duckdb --endpoint=/v1/completions --model=gpt-4
```

## Calibration

The Calibration module provides tools for calibrating simulation models to match real-world performance. It supports:

- Parameter discovery for finding optimal simulation parameters
- Cross-validation for ensuring calibration robustness
- Uncertainty quantification for understanding calibration confidence

## Reporting

The Reporting module generates reports on simulation validation results, including:

- Comparative reports showing simulated vs. real-world performance
- Technical reports with detailed validation metrics
- Executive summaries for high-level insights

## Integration with API Management UI

The Simulation Validation module integrates with the API Management UI to provide real-time validation of API metrics and recommendations. The integration supports:

1. Database-backed storage of API metrics, predictions, anomalies, and recommendations
2. Validation of data quality and prediction accuracy
3. Visualization of validation results
4. Recommendations for improving API performance and reliability