# Time-Series Performance Tracking Guide

**Date: March 25, 2025**  
**Status: Implementation Complete (100%)**

This guide provides comprehensive information about the Time-Series Performance Tracking system for the IPFS Accelerate Python Framework. This system allows tracking performance metrics over time, detecting regressions, analyzing trends, and generating visualizations and reports.

## Overview

The Time-Series Performance Tracking system provides:

1. **Versioned Test Results**: Store performance metrics with git version and environment information
2. **Regression Detection**: Automatically detect performance degradations based on configurable thresholds
3. **Trend Analysis**: Analyze performance trends over time with statistical methods
4. **Visualization**: Generate charts and visualizations of performance metrics over time
5. **Notification System**: Send alerts for detected regressions through various channels
6. **Reporting**: Generate comprehensive performance reports in Markdown or HTML format

## Quick Start

```bash
# Run a quick test with sample data
python test/run_time_series_performance.py --quick-test

# Run full test suite
python test/run_time_series_performance.py --full-test

# Use a specific database path
python test/run_time_series_performance.py --quick-test --db-path ./my_benchmark_db.duckdb
```

## Command-Line Usage

The time-series performance tracking system includes a command-line interface for common operations:

```bash
# Record a performance result
python test/time_series_performance.py record --model-id 1 --hardware-id 1 --batch-size 4 --throughput 125.7 --latency 8.2 --memory 1024 --power 180

# Set baselines for all model-hardware combinations
python test/time_series_performance.py baseline --all --days 7 --min-samples 3

# Detect regressions for a specific model
python test/time_series_performance.py regression --model-id 1 --days 14 --notify

# Analyze trends for all models and hardware
python test/time_series_performance.py trend --metric throughput --days 30 --visualize

# Generate a performance report
python test/time_series_performance.py report --days 30 --format markdown --output performance_report.md
```

## Python API Usage

The system provides a Python API for integration with other components:

```python
from time_series_performance import TimeSeriesPerformance

# Initialize with the database path
ts_perf = TimeSeriesPerformance(db_path='./benchmark_db.duckdb')

# Record a performance result
result_id = ts_perf.record_performance_result(
    model_id=1,
    hardware_id=1,
    batch_size=4,
    sequence_length=128,
    precision='fp32',
    throughput=125.7,
    latency=8.2,
    memory=1024,
    power=180
)

# Set baselines for all model-hardware combinations
baseline_results = ts_perf.set_all_baselines(days_lookback=7, min_samples=3)

# Detect regressions
regressions = ts_perf.detect_regressions(days_lookback=14)

# Analyze trends
trends = ts_perf.analyze_trends(metric='throughput', days_lookback=30, min_samples=5)

# Generate visualizations
viz_path = ts_perf.generate_trend_visualization(
    metric='throughput',
    days_lookback=30,
    output_path='throughput_trend.png'
)

# Generate a report
report_path = ts_perf.export_performance_report(
    days_lookback=30,
    format='markdown',
    output_path='performance_report.md'
)
```

## Database Schema

The time-series performance tracking system extends the existing benchmark database schema with the following tables:

1. **performance_baselines**: Stores baseline performance metrics for each model-hardware-config combination
2. **performance_regressions**: Tracks detected performance regressions
3. **performance_trends**: Stores trend analysis results
4. **regression_notifications**: Tracks notification status for detected regressions

The schema also adds versioning columns to the existing `performance_results` table:
- `version_tag`: Version tag for the code or model
- `git_commit_hash`: Git commit hash at the time of testing
- `environment_hash`: Hash representing the environment configuration
- `run_group_id`: ID to group related test runs

## Configuration

The system can be configured with the following options:

### Regression Thresholds

Thresholds for detecting regressions can be configured when initializing the `TimeSeriesPerformance` class:

```python
regression_thresholds = {
    'throughput': -5.0,  # 5% worse (negative because lower is worse)
    'latency': 5.0,      # 5% worse (positive because higher is worse)
    'memory': 5.0,       # 5% worse (positive because higher is worse)
    'power': 5.0         # 5% worse (positive because higher is worse)
}

ts_perf = TimeSeriesPerformance(
    db_path='./benchmark_db.duckdb',
    regression_thresholds=regression_thresholds
)
```

### Notification Configuration

The notification system can be configured to send alerts through various channels:

```python
notification_config = {
    'enabled': True,
    'methods': ['log', 'email', 'slack', 'github_issue', 'webhook'],
    'targets': {
        'email': ['team@example.com'],
        'slack': 'https://hooks.slack.com/services/...',
        'github': {
            'repo': 'org/repo',
            'token': 'github_token',
            'labels': ['regression', 'performance']
        },
        'webhook': 'https://example.com/webhook'
    }
}

ts_perf = TimeSeriesPerformance(
    db_path='./benchmark_db.duckdb',
    notification_config=notification_config
)
```

## Integration with CI/CD

The time-series performance tracking system can be integrated with CI/CD systems to automatically track performance across commits and branches:

```bash
# In CI pipeline
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# Run performance tests
python test/run_model_benchmarks.py --models bert-base-uncased,t5-small --hardware cuda

# Set baselines if needed
python test/time_series_performance.py baseline --all --days 7 --min-samples 3

# Detect regressions
python test/time_series_performance.py regression --days 7 --notify

# Generate and publish report
python test/time_series_performance.py report --days 30 --format html --output performance_report.html
```

## Best Practices

1. **Regular Baseline Updates**: Update performance baselines regularly to account for expected changes
2. **Environment Consistency**: Maintain consistent test environments to reduce noise in measurements
3. **Sufficient Samples**: Collect enough samples (at least 3-5) before setting baselines
4. **Threshold Tuning**: Adjust regression thresholds based on metric stability
5. **Regression Verification**: Verify detected regressions with additional tests before taking action
6. **Historical Data Retention**: Retain historical performance data for long-term trend analysis
7. **Regular Trend Analysis**: Run trend analysis regularly to identify gradual degradations

## API Reference

The `TimeSeriesPerformance` class provides the following main methods:

### Recording Performance

- `record_performance_result()`: Record a performance result with versioning information
- `get_environment_hash()`: Generate a hash representing the current environment

### Baseline Management

- `set_baseline()`: Set a performance baseline for a model-hardware-config combination
- `set_all_baselines()`: Set baselines for all model-hardware-config combinations

### Regression Detection

- `detect_regressions()`: Detect performance regressions based on configured thresholds

### Trend Analysis

- `analyze_trends()`: Analyze performance trends over time
- `get_performance_history()`: Get historical performance data as a DataFrame

### Visualization and Reporting

- `generate_trend_visualization()`: Generate visualization of performance trends
- `export_performance_report()`: Export a performance report with trends and regressions

### Notification

- `_send_regression_notifications()`: Send notifications for detected regressions
- `_send_log_notification()`, `_send_email_notification()`, `_send_slack_notification()`, 
  `_send_github_issue()`, `_send_webhook_notification()`: Send notifications via different channels

## Troubleshooting

### Common Issues

1. **Schema Errors**: Make sure the time-series schema is applied to the database
2. **Not Enough Samples**: Ensure enough performance samples are collected before setting baselines
3. **False Positives**: Adjust regression thresholds if too many false positives are detected
4. **Database Connection Errors**: Verify database path and permissions
5. **Visualization Errors**: Make sure matplotlib and other dependencies are installed
6. **Notification Issues**: Check notification configuration and credentials

## Schema Migration

If you are migrating from the previous database schema, run the following SQL script:

```bash
python -c "import duckdb; conn = duckdb.connect('benchmark_db.duckdb'); conn.execute(open('test/db_schema/time_series_schema.sql').read())"
```