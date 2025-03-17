# Result Aggregator System

This directory contains the Result Aggregator System for the Distributed Testing Framework. The system collects, processes, analyzes, and visualizes test results from distributed worker nodes.

## Directory Structure

- `integrated_analysis_system.py`: Main unified interface for the entire result aggregation and analysis system
- `service.py`: Core service for storing, processing, and analyzing test results
- `coordinator_integration.py`: Integration with the Coordinator for real-time result processing
- `analysis/`: Advanced statistical analysis components
  - `analysis.py`: Statistical analysis of performance metrics, including workload distribution, failure patterns, circuit breaker analysis, and time series forecasting
- `ml_detection/`: Machine learning components for anomaly detection
  - `ml_anomaly_detector.py`: ML-based anomaly detection implementation
- `pipeline/`: Data processing pipeline framework
  - `pipeline.py`: Core pipeline framework for data processing
  - `transforms.py`: Transform classes for data processing pipelines
- `transforms/`: General data transformation utilities
  - `transforms.py`: Data transformation utilities

## Functionality

The Result Aggregator System provides the following functionality:

1. **Integrated Analysis**: Unified interface for all result aggregation and analysis features
2. **Real-time Analysis**: Continuous monitoring and analysis of test results as they arrive
3. **Result Aggregation**: Collect and store test results from distributed worker nodes
4. **Statistical Analysis**: Analyze performance metrics to identify trends and patterns
5. **Multi-dimensional Analysis**: Compare performance across different dimensions (hardware, model type, etc.)
6. **Anomaly Detection**: Use machine learning to detect anomalies in test results
7. **Failure Pattern Detection**: Identify recurring failure patterns and correlations
8. **Workload Distribution Analysis**: Analyze the distribution of tasks across worker nodes
9. **Circuit Breaker Analysis**: Evaluate the effectiveness of circuit breaker implementations
10. **Visualization**: Generate interactive visualizations for various analysis types
11. **Performance Forecasting**: Predict future performance metrics with confidence intervals
12. **Reporting**: Generate comprehensive analysis reports in Markdown, HTML, and JSON formats
13. **Notification**: Send notifications when anomalies or significant trends are detected
14. **Time Series Analysis**: Track performance trends over time
15. **Data Transformation**: Process result data through customizable pipelines

## Usage

### Basic Usage

```python
from result_aggregator.integrated_analysis_system import IntegratedAnalysisSystem

# Initialize the system
analysis_system = IntegratedAnalysisSystem(db_path='./benchmark_db.duckdb')

# Register with a coordinator (optional)
analysis_system.register_with_coordinator(coordinator)

# Store a test result
result_id = analysis_system.store_result(test_result)

# Analyze results
analysis_results = analysis_system.analyze_results(
    filter_criteria={'test_type': 'benchmark'},
    analysis_types=['trends', 'anomalies', 'patterns']
)

# Generate comprehensive report
report = analysis_system.generate_report(
    analysis_results=analysis_results,
    format='html',
    output_path='report.html'
)

# Generate visualizations
analysis_system.visualize_results(
    visualization_type='trends',
    data=analysis_results.get('trends'),
    metrics=['throughput', 'latency'],
    output_path='visualizations/trends.png'
)

# Close the system when done
analysis_system.close()
```

### Command Line Usage

The IntegratedAnalysisSystem also provides a command-line interface:

```bash
# Analyze results
python -m result_aggregator.integrated_analysis_system --analyze --days 30 --filter '{"test_type": "benchmark"}'

# Generate report
python -m result_aggregator.integrated_analysis_system --report --report-type comprehensive --format html --output report.html

# Generate visualization
python -m result_aggregator.integrated_analysis_system --visualize --viz-type trends --viz-output trends.png

# Clean up old data
python -m result_aggregator.integrated_analysis_system --cleanup --keep-days 90
```

## Documentation

For detailed documentation, see the [Result Aggregation Guide](../docs/RESULT_AGGREGATION_GUIDE.md).

## Examples

For a complete example of using the Result Aggregator with the Coordinator, see [result_aggregator_example.py](../examples/result_aggregator_example.py).

## Testing

To run the tests for the Result Aggregator:

```bash
# Run unit tests
python -m unittest ../tests/test_integrated_analysis_system.py

# Run all tests and examples
../run_integrated_analysis_tests.sh
```

## Integration

The Result Aggregator integrates with the Coordinator through the `integrated_analysis_system.py` and `coordinator_integration.py` modules, which provide real-time processing of test results as they are received from worker nodes.

### Notification System

The IntegratedAnalysisSystem includes a notification system for alerting users to anomalies and significant performance trends:

```python
# Register notification handler
def notification_handler(notification):
    print(f"Notification: {notification['message']}")

analysis_system.register_notification_handler(notification_handler)
```

## Database Schema

The Result Aggregator uses the following database tables:

- `test_results`: Stores basic information about each test result
- `performance_metrics`: Stores individual performance metrics for each test result
- `anomaly_detections`: Stores detected anomalies
- `result_aggregations`: Stores cached aggregation results
- `analysis_reports`: Stores generated analysis reports
- `circuit_breaker_stats`: Stores circuit breaker performance data
- `failure_patterns`: Stores detected failure patterns
- `workload_distribution`: Stores workload distribution analytics

See the [Result Aggregation Guide](../docs/RESULT_AGGREGATION_GUIDE.md) for detailed schema information.

## Dependencies

The Result Aggregator has the following dependencies:

- **Required**: Python 3.7+, DuckDB
- **Optional**: pandas, numpy, matplotlib, seaborn, scipy

Optional dependencies enable additional features:
- Data analysis (pandas, numpy): Advanced statistical analysis
- Visualization (matplotlib, seaborn): Generate visualizations
- Statistical analysis (scipy): Advanced statistical tests and forecasting
- ML detection (sklearn): Machine learning-based anomaly detection