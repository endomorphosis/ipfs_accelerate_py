# Result Aggregation System Guide

This guide explains how to use the Result Aggregation System with the Distributed Testing Framework. The Result Aggregation System collects, processes, analyzes, and visualizes test results from distributed worker nodes, providing valuable insights into testing performance, anomalies, and trends.

## Overview

The Result Aggregation System consists of the following components:

1. **Integrated Analysis System**: Main entry point and unified interface for all result aggregation and analysis features.
2. **Result Aggregator Service**: Core service for storing, processing, and analyzing test results.
3. **Coordinator Integration**: Connects the Result Aggregator Service with the Coordinator to capture test results in real-time.
4. **Analysis Module**: Provides advanced statistical analysis of test results, failure patterns, circuit breaker performance, workload distribution, and time series forecasting.
5. **ML Anomaly Detection**: Uses machine learning to detect anomalies in test results.
6. **Pipeline Framework**: Processes test results through configurable data transformation pipelines.
7. **Visualization Module**: Creates static and interactive visualizations of test results.
8. **Performance Analyzer**: Provides comprehensive performance analysis including regression detection, hardware comparison, and resource efficiency analysis.
9. **Web Dashboard**: Provides a user-friendly web interface for exploring and analyzing test results.

## Architecture

The system uses a layered architecture:

```
                                                   +----------------------+
                                                   | Web Dashboard        |
                                                   | - Interactive UI     |
                                                   | - REST API           |
                                                   | - Real-time Updates  |
                                                   +----------^-----------+
                                                              |
                                                              |
+-------------------+     +-------------------------+     +----------------------+
| Coordinator       |     | Result Aggregator       |     | Analysis Components      |
| - Task Execution  |     | Integration             |     | - Statistical Analysis    |
| - Worker Management|<-->| - Real-time Processing  |<-->| - ML Anomaly Detection    |
| - Task Scheduling |     | - Database Integration  |     | - Trend Detection        |
| - Result Collection|    | - Notification System   |     | - Performance Regression |
|                   |    |                         |     | - Hardware Comparison    |
|                   |    |                         |     | - Resource Efficiency    |
|                   |    |                         |     | - Visualization          |
+-------------------+     +-------------------------+     +----------------------+
                                       |
                                       v
                          +-----------------------------+
                          | DuckDB Database             |
                          | - Test Results              |
                          | - Performance Metrics       |
                          | - Anomaly Detections        |
                          | - Analysis Reports          |
                          +-----------------------------+
```

## Installation and Setup

The Result Aggregation System is integrated with the Distributed Testing Framework and shares the same database. No additional installation is required beyond the standard framework dependencies.

Required dependencies:
- Python 3.8+
- DuckDB
- NumPy
- Pandas
- Matplotlib (optional, for static visualization)
- Plotly (optional, for interactive visualization)
- scikit-learn (optional, for ML-based anomaly detection)
- Flask (optional, for web dashboard)
- Flask-CORS (optional, for web dashboard)
- Flask-SocketIO (optional, for real-time updates in web dashboard)

## Integration with Coordinator

There are two ways to integrate the Result Aggregator with the Coordinator:

### Method 1: Using the Integrated Analysis System (Recommended)

The recommended approach uses the new Integrated Analysis System as a unified interface:

```python
from coordinator import DistributedTestingCoordinator
from result_aggregator.integrated_analysis_system import IntegratedAnalysisSystem

# Initialize the coordinator
coordinator = DistributedTestingCoordinator(
    db_path="./benchmark_db.duckdb",
    # Other coordinator options...
)

# Initialize the integrated analysis system
analysis_system = IntegratedAnalysisSystem(
    db_path="./benchmark_db.duckdb",
    enable_ml=True,
    enable_visualization=True,
    enable_real_time_analysis=True,
    analysis_interval=timedelta(minutes=5)
)

# Register with coordinator
analysis_system.register_with_coordinator(coordinator)

# Register notification handler (optional)
analysis_system.register_notification_handler(my_notification_handler)
```

### Method 2: Using the Legacy Coordinator Integration

You can also use the original approach with ResultAggregatorIntegration:

```python
from coordinator import DistributedTestingCoordinator
from result_aggregator.coordinator_integration import ResultAggregatorIntegration

# Initialize the coordinator
coordinator = DistributedTestingCoordinator(
    db_path="./benchmark_db.duckdb",
    # Other coordinator options...
)

# Initialize the result aggregator integration
integration = ResultAggregatorIntegration(
    coordinator=coordinator,
    enable_ml=True,
    enable_visualization=True,
    enable_real_time_analysis=True
)

# Register with coordinator
integration.register_with_coordinator()

# Register notification callback (optional)
integration.register_notification_callback(my_notification_handler)
```

Both methods will automatically intercept task results and store them in the Result Aggregator Service for analysis, but the Integrated Analysis System provides a more unified and comprehensive interface.

## Core Features

### Real-time Result Processing

The system processes test results in real-time as they are received from worker nodes. Results are stored in the database and can be analyzed for anomalies immediately.

### Performance Trend Analysis

Analyze performance trends over time to identify improvements or regressions:

```python
# Get the Result Aggregator Service
service = integration.get_service()

# Analyze trends over the past hour
trends = service.analyze_performance_trends(
    filter_criteria={"start_time": "2025-03-15T00:00:00"}
)

# Process trend data
for metric, trend_data in trends.items():
    print(f"Metric: {metric}")
    print(f"Trend: {trend_data['trend']}")  # increasing, decreasing, or stable
    print(f"Percent Change: {trend_data['percent_change']}%")
```

### Anomaly Detection

The system uses machine learning algorithms to detect anomalies in test results:

```python
# Detect anomalies in recent results
anomalies = service.detect_anomalies(
    filter_criteria={"start_time": "2025-03-15T00:00:00"}
)

# Process detected anomalies
for anomaly in anomalies:
    print(f"Anomaly Score: {anomaly['score']}")
    print(f"Anomaly Type: {anomaly['type']}")
    print(f"Details: {anomaly['details']}")
```

### Performance Analysis

The Performance Analyzer provides comprehensive analysis capabilities for test results, including regression detection, hardware comparison, resource efficiency analysis, and time-based performance trend analysis.

#### Performance Regression Detection

Detect statistically significant regressions in performance metrics:

```python
# Detect performance regressions for a specific metric
regression_analysis = service.analyze_performance_regression(
    metric_name="throughput",
    baseline_period="7d",    # 7 days for baseline
    comparison_period="1d"   # 1 day for comparison
)

# Process regression results
if "metrics" in regression_analysis and "throughput" in regression_analysis["metrics"]:
    result = regression_analysis["metrics"]["throughput"]
    
    if result["status"] == "regression":
        print(f"Regression detected! Severity: {result['severity']}")
        print(f"Percent change: {result['percent_change']:.2f}%")
        print(f"Statistically significant: {result['is_statistically_significant']}")
        
        # Compare baseline vs current data
        print(f"Baseline mean: {result['baseline']['mean']:.2f}")
        print(f"Current mean: {result['comparison']['mean']:.2f}")
```

The regression analysis includes statistical significance testing (t-test) to reduce false positives, and classifies regressions into severity levels (critical, major, minor) based on configurable thresholds.

#### Hardware Performance Comparison

Compare performance across different hardware profiles:

```python
# Compare hardware performance for specific metrics
hardware_comparison = service.compare_hardware_performance(
    metrics=["throughput", "latency_ms"],
    test_type="benchmark",
    time_period="30d"
)

# Process hardware comparison results
if "summary" in hardware_comparison:
    print(f"Best overall hardware: {hardware_comparison['summary']['best_overall_hardware']}")
    
    # Best hardware by metric
    for metric, hw in hardware_comparison['summary']['best_by_metric'].items():
        print(f"Best hardware for {metric}: {hw}")
    
    # Hardware scores
    for hw, score in hardware_comparison['summary']['hardware_scores'].items():
        print(f"Hardware {hw} score: {score:.2f}")
```

The hardware comparison evaluates performance metrics across different hardware configurations and produces a weighted score to determine the optimal hardware for specific workloads.

#### Resource Efficiency Analysis

Analyze resource efficiency metrics to optimize performance-to-resource ratios:

```python
# Analyze resource efficiency
efficiency_analysis = service.analyze_resource_efficiency(
    test_type="benchmark",
    time_period="30d"
)

# Process efficiency analysis results
if "summary" in efficiency_analysis:
    print(f"Most efficient setup: {efficiency_analysis['summary']['most_efficient_setup']}")
    
    # Efficiency metrics by configuration
    for config_str, data in efficiency_analysis['efficiency_metrics'].items():
        print(f"Configuration: {config_str}")
        
        # Efficiency metrics
        for metric, value in data['efficiency_metrics'].items():
            print(f"{metric}: {value:.4f}")
```

Resource efficiency analysis examines metrics like throughput-per-memory, throughput-per-power, and time-memory efficiency to identify optimal configurations for resource-constrained environments.

#### Time-based Performance Analysis

Analyze performance trends over time with regression analysis and forecasting:

```python
# Analyze performance trends over time
time_analysis = service.analyze_performance_over_time(
    metric_name="throughput",
    grouping="day",
    time_period="90d"
)

# Process time-based analysis results
if "analysis" in time_analysis:
    analysis = time_analysis["analysis"]
    print(f"Trend: {analysis.get('trend', 'unknown')}")
    print(f"Best fitting model: {analysis.get('best_model', 'unknown')}")
    
    # Linear regression details
    if "linear_regression" in analysis:
        lr = analysis["linear_regression"]
        print(f"Linear model: y = {lr['slope']:.4f}x + {lr['intercept']:.4f} (R² = {lr['r2']:.4f})")
    
    # Forecast
    if "forecast" in analysis:
        print(f"Forecast (next 3 points): {', '.join([f'{v:.2f}' for v in analysis['forecast']])}")
```

The time-based analysis uses linear and polynomial regression to identify trends and forecast future performance, helping predict potential issues before they occur.

#### Comprehensive Performance Reporting

Generate detailed performance reports in various formats:

```python
# Generate a comprehensive performance report in Markdown format
report = service.generate_performance_report(
    report_type="comprehensive",  # Options: comprehensive, regression, hardware_comparison, efficiency, time_analysis
    format="markdown",            # Options: markdown, html, json
    time_period="30d"
)

# Save the report to a file
with open("performance_report.md", "w") as f:
    f.write(report)

# Generate an HTML report for better visualization
html_report = service.generate_performance_report(
    report_type="comprehensive",
    format="html",
    time_period="30d"
)

# Save the HTML report
with open("performance_report.html", "w") as f:
    f.write(html_report)
```

The performance reports include detailed analysis of all aspects of performance with visualizations and actionable insights to help identify opportunities for optimization.

### Report Generation

Generate comprehensive analysis reports in various formats:

```python
# Generate a performance report in Markdown format
report = service.generate_analysis_report(
    report_type="performance",
    format="markdown"
)

# Save the report to a file
with open("performance_report.md", "w") as f:
    f.write(report)

# Generate an anomaly report in HTML format
anomaly_report = service.generate_analysis_report(
    report_type="anomaly",
    format="html"
)

# Save the HTML report
with open("anomaly_report.html", "w") as f:
    f.write(anomaly_report)
    
# Generate a CSV report for data analysis in spreadsheets
csv_report = service.generate_analysis_report(
    report_type="performance",
    format="csv"
)

# Save the CSV report
with open("performance_report.csv", "w") as f:
    f.write(csv_report)
```

### Notification System

The Result Aggregator can notify interested parties when anomalies or significant trends are detected:

```python
# Define a notification handler
def my_notification_handler(notification):
    """Handle notifications from the Result Aggregator"""
    notification_type = notification["type"]
    severity = notification["severity"]
    message = notification["message"]
    details = notification["details"]
    
    # Process notification based on type and severity
    if notification_type == "anomaly" and severity == "warning":
        # Send alert to team
        send_alert(message, details)
    elif notification_type == "trend" and severity == "info":
        # Log trend information
        log_trend(message, details)

# Register the notification handler
integration.register_notification_callback(my_notification_handler)
```

## Advanced Usage

### Custom Analysis Pipelines

For advanced data processing, you can use the Pipeline Framework:

```python
from result_aggregator.pipeline.pipeline import DataSource, ProcessingPipeline
from result_aggregator.pipeline.transforms import (
    FilterTransform, TimeWindowTransform, AggregateTransform,
    PivotTransform, CalculatedMetricTransform
)

# Create data source
data_source = DataSource(service.db, "performance_metrics", "test_results")

# Create pipeline with transforms
pipeline = ProcessingPipeline(
    transforms=[
        # Filter by test_type
        FilterTransform(conditions={"test_type": "benchmark"}),
        # Filter by time window
        TimeWindowTransform(days=7),
        # Group by metric and calculate statistics
        AggregateTransform(
            group_by=["metric_name", "day"],
            aggregations={"value": ["mean", "median", "min", "max"]}
        ),
        # Pivot by day
        PivotTransform(
            index=["metric_name"],
            columns="day",
            values="value_mean"
        ),
        # Calculate change over time
        CalculatedMetricTransform(
            "percent_change",
            lambda df: (df.iloc[:, -1] - df.iloc[:, 0]) / df.iloc[:, 0] * 100
        )
    ]
)

# Execute pipeline
result = pipeline.execute(data_source)
```

### Using the Integrated Analysis System

The Integrated Analysis System provides a unified interface for all Result Aggregator features:

```python
from result_aggregator.integrated_analysis_system import IntegratedAnalysisSystem

# Initialize the system
analysis_system = IntegratedAnalysisSystem(
    db_path="./benchmark_db.duckdb",
    enable_ml=True,
    enable_visualization=True,
    enable_real_time_analysis=True
)

# Store a test result
result = {
    "task_id": "task_123",
    "worker_id": "worker_456",
    "type": "benchmark",
    "status": "success",
    "metrics": {
        "throughput": 145.7,
        "latency": 6.8,
        "memory_usage": 1024.5
    },
    "details": {
        "model": "bert",
        "hardware": "cuda",
        "batch_size": 8,
        "precision": "fp16"
    }
}
result_id = analysis_system.store_result(result)

# Perform comprehensive analysis
analysis_results = analysis_system.analyze_results(
    filter_criteria={"test_type": "benchmark"},
    analysis_types=["trends", "anomalies", "workload", "failures", "performance", "circuit_breaker", "recovery", "forecast"],
    metrics=["throughput", "latency", "memory_usage"],
    group_by="hardware",
    time_period_days=30
)

# Generate a comprehensive report
report = analysis_system.generate_report(
    analysis_results=analysis_results,
    report_type="comprehensive",
    format="markdown",
    output_path="reports/comprehensive_report.md"
)

# Generate visualizations
analysis_system.visualize_results(
    visualization_type="trends",
    data=analysis_results.get("trends"),
    metrics=["throughput", "latency"],
    output_path="visualizations/performance_trends.png"
)

analysis_system.visualize_results(
    visualization_type="workload_distribution",
    data=analysis_results.get("workload_distribution"),
    output_path="visualizations/workload_distribution.png"
)

analysis_system.visualize_results(
    visualization_type="failure_patterns",
    data=analysis_results.get("failure_patterns"),
    output_path="visualizations/failure_patterns.png"
)

# Register a notification handler
def notification_handler(notification):
    print(f"Notification: {notification['type']} - {notification['severity']}")
    print(f"Message: {notification['message']}")
    
analysis_system.register_notification_handler(notification_handler)

# Close the system when done
analysis_system.close()
```

### Data Transformations

Use the data transformation utilities for preprocessing data:

```python
from result_aggregator.transforms.transforms import (
    normalize_metrics, extract_features, clean_outliers, 
    transform_for_analysis, prepare_for_visualization
)

# Get data from database
query = """
SELECT m.metric_name, m.metric_value, t.test_type, t.timestamp
FROM performance_metrics m
JOIN test_results t ON m.result_id = t.id
WHERE t.status = 'completed'
"""
df = pd.read_sql(query, service.db)

# Normalize metrics
normalized_df = normalize_metrics(df, method="z-score")

# Extract features for analysis
features_df = extract_features(df, feature_columns=["metric_value", "execution_time"])

# Clean outliers
cleaned_df = clean_outliers(df, method="z-score", threshold=3.0)

# Prepare for visualization
viz_data = prepare_for_visualization(
    df, visualization_type="time_series", 
    x_column="timestamp", y_column="metric_value", 
    group_column="metric_name"
)
```

### Periodic Batch Analysis

Run scheduled analysis to detect long-term trends and recurring patterns:

```python
import asyncio
from datetime import datetime, timedelta

async def run_periodic_analysis():
    """Run analysis on a schedule"""
    while True:
        # Get results from the past 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        filter_criteria = {"start_time": yesterday.isoformat()}
        
        # Analyze performance trends
        trends = service.analyze_performance_trends(filter_criteria)
        
        # Detect anomalies
        anomalies = service.detect_anomalies(filter_criteria)
        
        # Generate and save daily report
        report = service.generate_analysis_report(
            report_type="performance",
            filter_criteria=filter_criteria,
            format="markdown"
        )
        
        # Save report
        report_path = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        # Wait 24 hours before next analysis
        await asyncio.sleep(24 * 60 * 60)

# Start periodic analysis
asyncio.create_task(run_periodic_analysis())
```

## Web Dashboard

The Result Aggregator includes a web-based dashboard for visualizing and interacting with test results. The dashboard provides a user-friendly interface for exploring test results, performance trends, and anomalies.

### Starting the Web Dashboard

To start the web dashboard, use the provided run script:

```bash
# Start with default settings
python run_web_dashboard.py

# Specify custom port and database path
python run_web_dashboard.py --port 8050 --db-path ./test_results.duckdb

# Run in debug mode
python run_web_dashboard.py --debug
```

Once started, access the dashboard in your web browser at `http://localhost:8050`.

### Dashboard Features

The web dashboard includes the following features:

1. **Main Dashboard**: Overview of key metrics, recent test results, and performance trends.
2. **Test Results**: Detailed view of all test results with filtering and pagination.
3. **Performance Trends**: Visualization of performance trends over time for different metrics.
4. **Anomalies**: View and analyze detected anomalies with detailed information.
5. **Reports**: Generate and view analysis reports in different formats.
6. **Real-time Updates**: Receive notifications for new results and anomalies.

### API Endpoints

The web dashboard provides a comprehensive REST API for programmatic access to data:

```bash
# Get test results with filtering
curl "http://localhost:8050/api/results?test_type=benchmark&status=completed&limit=10"

# Get aggregated results
curl "http://localhost:8050/api/aggregated?aggregation_type=mean&group_by=test_type,worker_id"

# Get performance trends
curl "http://localhost:8050/api/trends?metrics=throughput,latency"

# Generate a performance report in Markdown format
curl "http://localhost:8050/api/report?report_type=performance&format=markdown"

# Generate a performance report in CSV format for spreadsheet analysis
curl "http://localhost:8050/api/report?report_type=performance&format=csv"
```

For a complete list of API endpoints, see the [Web Dashboard Guide](WEB_DASHBOARD_GUIDE.md).

### Authentication

The web dashboard includes basic authentication to secure access:

- Default username: `admin`
- Default password: `admin_password`

You should change these credentials for production use.

### Visualization Types

The dashboard supports various visualization types:

1. **Performance Charts**: Line charts showing performance metrics over time.
2. **Trend Analysis**: Bar charts and line charts showing performance trends.
3. **Anomaly Dashboard**: Scatter plots, box plots, and other visualizations for anomaly detection.
4. **Summary Dashboard**: Various charts and tables showing summary statistics.

For more details about the web dashboard, refer to the [Web Dashboard Guide](WEB_DASHBOARD_GUIDE.md).

## Examples

See the following example scripts:
- `examples/result_aggregator_example.py` - Example of using the Result Aggregator with the Coordinator
- `run_test_web_dashboard.py` - Demo script that generates sample data and starts the web dashboard

## Database Schema

The Result Aggregator uses the following database tables:

### test_results

Stores basic information about each test result:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| task_id | VARCHAR | Task ID from the coordinator |
| worker_id | VARCHAR | Worker ID that executed the task |
| timestamp | TIMESTAMP | When the result was created |
| test_type | VARCHAR | Type of test (benchmark, unit, integration, etc.) |
| status | VARCHAR | Status of the test (completed, failed) |
| duration | FLOAT | Execution time in seconds |
| details | JSON | Additional details about the test |
| metrics | JSON | Performance metrics as JSON |

### performance_metrics

Stores individual performance metrics for each test result:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| result_id | INTEGER | Foreign key to test_results |
| metric_name | VARCHAR | Name of the metric |
| metric_value | FLOAT | Value of the metric |
| metric_unit | VARCHAR | Unit of the metric |

### anomaly_detections

Stores detected anomalies:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| result_id | INTEGER | Foreign key to test_results |
| detection_time | TIMESTAMP | When the anomaly was detected |
| anomaly_score | FLOAT | Score indicating anomaly severity |
| anomaly_type | VARCHAR | Type of anomaly |
| is_confirmed | BOOLEAN | Whether the anomaly is confirmed |
| details | JSON | Additional details about the anomaly |

### result_aggregations

Stores cached aggregation results:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| aggregation_name | VARCHAR | Name of the aggregation |
| aggregation_type | VARCHAR | Type of aggregation |
| filter_criteria | JSON | Criteria used for filtering |
| aggregation_data | JSON | Aggregated data |
| created_at | TIMESTAMP | When the aggregation was created |
| updated_at | TIMESTAMP | When the aggregation was last updated |

### analysis_reports

Stores generated analysis reports:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| report_name | VARCHAR | Name of the report |
| report_type | VARCHAR | Type of report |
| filter_criteria | JSON | Criteria used for filtering |
| report_data | JSON | Report data |
| created_at | TIMESTAMP | When the report was created |

### circuit_breaker_stats

Stores circuit breaker performance data:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| result_id | INTEGER | Foreign key to test_results |
| circuit_breaker_state | VARCHAR | State of the circuit breaker (closed, open, half_open) |
| failure_count | INTEGER | Current failure count |
| success_streak | INTEGER | Current success streak |
| failure_threshold | INTEGER | Threshold for opening the circuit |
| recovery_timeout | FLOAT | Timeout in seconds before half-open state |
| transitions | JSON | History of state transitions |
| created_at | TIMESTAMP | When the record was created |

### failure_patterns

Stores detected failure patterns:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| pattern_name | VARCHAR | Name of the pattern |
| pattern_type | VARCHAR | Type of pattern (worker, test_type, time) |
| detection_time | TIMESTAMP | When the pattern was detected |
| confidence | FLOAT | Confidence in the pattern (0.0 to 1.0) |
| related_results | JSON | IDs of related test results |
| pattern_data | JSON | Data describing the pattern |
| created_at | TIMESTAMP | When the record was created |

### workload_distribution

Stores workload distribution analytics:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| analysis_time | TIMESTAMP | When the analysis was performed |
| time_period | VARCHAR | Time period for the analysis (e.g., 7d, 30d) |
| worker_stats | JSON | Statistics per worker |
| distribution_stats | JSON | Overall distribution statistics |
| gini_coefficient | FLOAT | Measure of distribution inequality (0.0 to 1.0) |
| created_at | TIMESTAMP | When the record was created |

## Advanced Configuration

### Machine Learning Configuration

The ML component can be configured to use different algorithms for anomaly detection:

```python
# Configure isolation forest parameters
service.service.ml_models["isolation_forest"]["model"].set_params(
    contamination=0.1,  # Adjust contamination parameter (0.0 to 0.5)
    n_estimators=200    # Increase number of estimators for better accuracy
)

# Configure Z-score threshold for statistical anomaly detection
service.service._z_score_threshold = 3.5  # Default is 3.0
```

### Performance and Optimization

For large datasets, consider these optimization options:

```python
# Set analysis interval to reduce load on the system
integration.analysis_interval = timedelta(minutes=15)  # Default is 5 minutes

# Clean up old data to maintain performance
# Keep data for the last 90 days
service.cleanup_old_data(days=90)
```

## Troubleshooting

### Common Issues

1. **High CPU or Memory Usage**
   - Increase the analysis interval
   - Reduce the amount of data being processed by adjusting filter criteria
   - Clean up old data regularly

2. **Missing or Incomplete Results**
   - Check database connection
   - Verify coordinator hooks are properly registered
   - Check error logs for exceptions during result processing

3. **ML Anomaly Detection Not Working**
   - Ensure scikit-learn is installed
   - Check if enough data is available (at least 10 samples per test type)
   - Adjust parameters for sensitivity

### Logging

The Result Aggregator logs detailed information that can help diagnose issues:

```python
import logging

# Set logging level for the Result Aggregator
logging.getLogger("result_aggregator").setLevel(logging.DEBUG)

# To log to a file in addition to console
file_handler = logging.FileHandler("result_aggregator.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger("result_aggregator").addHandler(file_handler)
```

## Performance Metrics

The Result Aggregator automatically tracks these standard metrics:

1. **Execution Time**: Duration of task execution in seconds
2. **Throughput**: Items processed per second (if available)
3. **Latency**: Response time in milliseconds (if available)
4. **Memory Usage**: Memory consumption in MB (if available)
5. **CPU Usage**: CPU utilization percentage (if available)
6. **Error Rate**: Percentage of failed tasks

Custom metrics can be included in task results and will be automatically tracked.

## Conclusion

The Result Aggregation System provides comprehensive capabilities for collecting, analyzing, and visualizing test results from the Distributed Testing Framework. By integrating with the Coordinator, it enables real-time analysis of test performance and automatic detection of anomalies and trends. The web dashboard provides a user-friendly interface for exploring and analyzing test results, making it accessible to team members without requiring programming knowledge.

## Command Line Interface

The Integrated Analysis System provides a command-line interface for common operations:

```bash
# Analyze results for the past 30 days
python -m result_aggregator.integrated_analysis_system --analyze --days 30

# Analyze with specific filter criteria (JSON format)
python -m result_aggregator.integrated_analysis_system --analyze --filter '{"test_type": "benchmark", "status": "success"}'

# Generate a comprehensive report in markdown format
python -m result_aggregator.integrated_analysis_system --report --report-type comprehensive --format markdown --output report.md

# Generate a performance-focused report in HTML format
python -m result_aggregator.integrated_analysis_system --report --report-type performance --format html --output performance_report.html

# Generate a visualization of performance trends
python -m result_aggregator.integrated_analysis_system --visualize --viz-type trends --viz-output trends.png

# Generate a visualization of workload distribution
python -m result_aggregator.integrated_analysis_system --visualize --viz-type workload_distribution --viz-output workload.png

# Generate a visualization of anomalies
python -m result_aggregator.integrated_analysis_system --visualize --viz-type anomalies --viz-output anomalies.png

# Clean up old data (keep last 90 days)
python -m result_aggregator.integrated_analysis_system --cleanup --keep-days 90
```

### Command Line Options

- **Analysis Options**:
  - `--analyze`: Perform analysis on test results
  - `--filter`: JSON filter criteria for analysis
  - `--days`: Number of days to look back for analysis (default: 30)

- **Report Options**:
  - `--report`: Generate report
  - `--report-type`: Type of report to generate (comprehensive, performance, summary, anomaly)
  - `--format`: Format of the report (markdown, html, json)
  - `--output`: Path to save the report

- **Visualization Options**:
  - `--visualize`: Generate visualizations
  - `--viz-type`: Type of visualization (trends, time_series, anomalies, performance_comparison, workload_distribution, failure_patterns, circuit_breaker)
  - `--viz-output`: Path to save the visualization

- **Cleanup Options**:
  - `--cleanup`: Clean up old data
  - `--keep-days`: Number of days to keep when cleaning up old data (default: 90)

- **Database Options**:
  - `--db-path`: Path to DuckDB database file (default: ./benchmark_db.duckdb)

## Resources

For more detailed information, refer to the following resources:

- Example usage: `examples/result_aggregator_example.py` and `run_test_web_dashboard.py`
- Dashboard guide: `docs/WEB_DASHBOARD_GUIDE.md`
- API documentation: `result_aggregator/integrated_analysis_system.py`, `result_aggregator/service.py`, `result_aggregator/coordinator_integration.py`, and `result_aggregator/web_dashboard.py`
- Implementation details: `result_aggregator/analysis/`, `result_aggregator/ml_detection/`, `result_aggregator/pipeline/`, `result_aggregator/transforms/`, and `result_aggregator/visualization.py`