# Time-Series Performance Tracking Guide

**Date: March 15, 2025**  
**Version: 1.0**

This guide provides detailed information on using the Time-Series Performance Tracking system in the IPFS Accelerate Python Framework. This system allows you to track performance metrics over time, detect regressions, analyze trends, and generate comprehensive reports.

## Table of Contents

1. [Introduction](#introduction)
2. [System Components](#system-components)
3. [Setup](#setup)
4. [Recording Performance Data](#recording-performance-data)
5. [Setting Baselines](#setting-baselines)
6. [Detecting Regressions](#detecting-regressions)
7. [Analyzing Trends](#analyzing-trends)
8. [Generating Reports](#generating-reports)
9. [Integration with CI/CD](#integration-with-cicd)
10. [Command-Line Interface](#command-line-interface)
11. [Best Practices](#best-practices)

## Introduction

The Time-Series Performance Tracking system is designed to help you:

- Track performance metrics (throughput, latency, memory, power) over time
- Set performance baselines for different model-hardware combinations
- Detect performance regressions when metrics degrade
- Analyze performance trends over time
- Generate visualizations and reports
- Send notifications when regressions are detected

The system integrates with the DuckDB benchmark database and provides a comprehensive API and command-line interface for all these functions.

## System Components

The Time-Series Performance Tracking system consists of several key components:

1. **Database Schema**: Extended schema in DuckDB for storing time-series data, baselines, regressions, and trends
2. **TimeSeriesPerformance Class**: Python class that provides the API for all time-series tracking functions
3. **Command-Line Interface**: CLI for performing time-series operations directly from the command line
4. **Visualization Tools**: Utilities for generating charts and visualizations from time-series data
5. **Reporting System**: Tools for generating comprehensive performance reports
6. **Notification System**: Configurable system for sending regression alerts via various channels

## Setup

### Database Schema Setup

The time-series performance tracking system extends the existing benchmark database with additional tables and functions. To set up the schema:

```bash
# Make sure the db_schema directory exists
mkdir -p test/db_schema

# Apply the time-series schema to your database
python -c "import duckdb; conn = duckdb.connect('./benchmark_db.duckdb'); conn.execute(open('duckdb_api/schema/time_series_schema.sql').read())"
```

### Basic Usage

```python
from duckdb_api.core.time_series_performance import TimeSeriesPerformance

# Initialize with default database path
ts_perf = TimeSeriesPerformance()

# Or specify a database path
ts_perf = TimeSeriesPerformance(db_path="./my_benchmark_db.duckdb")

# Configure regression thresholds
ts_perf = TimeSeriesPerformance(
    regression_thresholds={
        'throughput': -5.0,  # 5% worse (negative because lower is worse)
        'latency': 5.0,      # 5% worse (positive because higher is worse)
        'memory': 5.0,       # 5% worse (positive because higher is worse)
        'power': 5.0         # 5% worse (positive because higher is worse)
    }
)

# Configure notifications
ts_perf = TimeSeriesPerformance(
    notification_config={
        'enabled': True,
        'methods': ['log', 'email', 'slack', 'github_issue'],
        'targets': {
            'email': ['team@example.com'],
            'slack': 'https://hooks.slack.com/services/XXX/YYY/ZZZ',
            'github': {
                'repo': 'organization/repository',
                'token': 'github_token',
                'labels': ['regression', 'performance']
            },
            'webhook': 'https://example.com/webhook'
        }
    }
)
```

## Recording Performance Data

The first step in time-series tracking is recording performance results. Each result includes metrics (throughput, latency, memory, power) along with configuration details (model, hardware, batch size, etc.).

```python
# Record a performance result
result_id = ts_perf.record_performance_result(
    model_id=1,                # ID of the model in the database
    hardware_id=2,             # ID of the hardware platform in the database
    batch_size=16,             # Batch size used in the test
    sequence_length=128,       # Sequence length or None
    precision="fp16",          # Precision format used
    throughput=1500.0,         # Throughput in items per second
    latency=10.5,              # Latency in milliseconds
    memory=2048.0,             # Memory usage in MB
    power=150.0,               # Power consumption in watts
    version_tag="v1.2.3",      # Optional version tag
    run_group_id="nightly_run" # Optional group ID for related runs
)

print(f"Recorded performance result with ID: {result_id}")
```

The system automatically captures additional metadata:
- Git commit hash (if available)
- Environment hash (system info, relevant environment variables)
- Timestamp

## Setting Baselines

Baselines establish reference points for performance metrics. They're computed as averages from recent results and used to detect regressions.

```python
# Set a baseline for a specific model-hardware-config combination
baseline_id = ts_perf.set_baseline(
    model_id=1,
    hardware_id=2,
    batch_size=16,
    sequence_length=128,
    precision="fp16",
    days_lookback=7,     # Number of days to look back for samples
    min_samples=3        # Minimum number of samples required
)

print(f"Set baseline with ID: {baseline_id}")

# Set baselines for all model-hardware-config combinations with enough samples
baseline_results = ts_perf.set_all_baselines(
    days_lookback=7,
    min_samples=3
)

print(f"Set {len([r for r in baseline_results if r['status'] == 'success'])} baselines")
```

Baselines are updated automatically when new results are added, ensuring your regression detection stays relevant.

## Detecting Regressions

Regression detection compares recent results against established baselines to identify performance degradation.

```python
# Detect regressions
regressions = ts_perf.detect_regressions(
    model_id=1,           # Optional: filter by model ID
    hardware_id=2,        # Optional: filter by hardware ID
    days_lookback=1       # Number of days to look back for results to compare
)

if regressions:
    print(f"Detected {len(regressions)} regressions:")
    for reg in regressions:
        print(f"  {reg['model_name']} on {reg['hardware_type']}: "
              f"{reg['regression_type']} degraded by {reg['severity']:.2f}%")
else:
    print("No regressions detected")
```

When regressions are detected, the system can send notifications via various channels (if configured):
- Log messages
- Email notifications
- Slack messages
- GitHub issues
- Webhook calls

## Analyzing Trends

Trend analysis examines performance metrics over time to identify patterns of improvement or degradation.

```python
# Analyze trends
trends = ts_perf.analyze_trends(
    model_id=1,            # Optional: filter by model ID
    hardware_id=2,         # Optional: filter by hardware ID
    metric="throughput",   # Metric to analyze: 'throughput', 'latency', 'memory', 'power'
    days_lookback=30,      # Number of days to look back for analysis
    min_samples=5          # Minimum number of samples required for analysis
)

if trends:
    print(f"Analyzed {len(trends)} trends:")
    for trend in trends:
        print(f"  {trend['model_name']} on {trend['hardware_type']}: "
              f"{trend['metric']} {trend['trend_direction']} by {trend['trend_magnitude']:.2f}% "
              f"(confidence: {trend['trend_confidence']:.2f})")
else:
    print("No significant trends detected")
```

Trend directions include:
- `improving`: Metrics are getting better over time
- `degrading`: Metrics are getting worse over time
- `stable`: Metrics are stable within a small range
- `volatile`: Metrics are changing but without a clear direction

## Generating Reports

The system can generate comprehensive performance reports in markdown or HTML formats.

```python
# Generate a performance report
report_path = ts_perf.export_performance_report(
    model_id=1,             # Optional: filter by model ID
    hardware_id=2,          # Optional: filter by hardware ID
    days_lookback=30,       # Number of days to include in the report
    format="markdown",      # Report format: 'markdown' or 'html'
    output_path="reports/performance_report.md"  # Output path for the report
)

print(f"Generated report at: {report_path}")
```

Reports include:
- Summary of regressions and trends
- Performance visualizations
- Detailed regression information
- Trend analysis results

### Visualizations

The system can also generate standalone visualizations of performance trends.

```python
# Generate a trend visualization
viz_path = ts_perf.generate_trend_visualization(
    model_id=1,
    hardware_id=2,
    metric="throughput",
    days_lookback=30,
    output_path="visualizations/throughput_trend.png"
)

print(f"Generated visualization at: {viz_path}")
```

## Integration with CI/CD

The Time-Series Performance Tracking system is designed to integrate with CI/CD pipelines for automated regression detection and reporting.

### GitHub Actions Integration Example

```yaml
name: Performance Regression Detection

on:
  workflow_run:
    workflows: ["Run Benchmarks"]
    types:
      - completed
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight

jobs:
  detect-regressions:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
        
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: Download benchmark database
        run: |
          aws s3 cp s3://benchmark-data/benchmark_db.duckdb ./benchmark_db.duckdb
          
      - name: Set baselines
        run: |
          python -m time_series_performance baseline --all --days 7 --min-samples 3
          
      - name: Detect regressions
        run: |
          python -m time_series_performance regression --notify
          
      - name: Generate report
        run: |
          python -m time_series_performance report --days 30 --format html --output reports/daily_report.html
          
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: performance-report
          path: reports/daily_report.html
```

## Command-Line Interface

The time-series performance tracking system includes a comprehensive command-line interface.

### Recording Performance Results

```bash
python -m time_series_performance record \
  --model-id 1 \
  --hardware-id 2 \
  --batch-size 16 \
  --sequence-length 128 \
  --precision fp16 \
  --throughput 1500.0 \
  --latency 10.5 \
  --memory 2048.0 \
  --power 150.0 \
  --version-tag v1.2.3 \
  --run-group nightly_run
```

### Setting Baselines

```bash
# Set a specific baseline
python -m time_series_performance baseline \
  --model-id 1 \
  --hardware-id 2 \
  --batch-size 16 \
  --sequence-length 128 \
  --precision fp16 \
  --days 7 \
  --min-samples 3

# Set all baselines
python -m time_series_performance baseline --all --days 7 --min-samples 3
```

### Detecting Regressions

```bash
# Detect regressions for all models and hardware
python -m time_series_performance regression --days 1 --notify

# Detect regressions for a specific model on a specific hardware
python -m time_series_performance regression \
  --model-id 1 \
  --hardware-id 2 \
  --days 1 \
  --throughput-threshold -10.0 \
  --latency-threshold 10.0 \
  --memory-threshold 8.0 \
  --power-threshold 8.0 \
  --notify
```

### Analyzing Trends

```bash
# Analyze trends
python -m time_series_performance trend \
  --model-id 1 \
  --hardware-id 2 \
  --metric throughput \
  --days 30 \
  --min-samples 5 \
  --visualize \
  --output throughput_trend.png
```

### Generating Reports

```bash
# Generate a performance report
python -m time_series_performance report \
  --model-id 1 \
  --hardware-id 2 \
  --days 30 \
  --format markdown \
  --output performance_report.md
```

## Best Practices

### Continuous Performance Tracking

For effective performance tracking:

1. **Record Results Consistently**: Consistently record benchmark results for all model-hardware combinations
2. **Set Baselines Regularly**: Update baselines regularly to account for expected performance changes
3. **Check for Regressions Daily**: Run regression detection daily to catch issues early
4. **Analyze Trends Monthly**: Perform trend analysis monthly to understand long-term patterns
5. **Generate Reports Automatically**: Automate report generation for stakeholders

### Regression Thresholds

Set appropriate regression thresholds based on your requirements:

- **Stable Production Systems**: Tighter thresholds (e.g., throughput: -3%, latency: 3%)
- **Development Environments**: Looser thresholds (e.g., throughput: -10%, latency: 10%)
- **Critical Components**: Very tight thresholds (e.g., throughput: -1%, latency: 1%)

### Sample Command Sequence

A typical workflow might include:

```bash
# Record benchmark results
python -m time_series_performance record --model-id 1 --hardware-id 2 ...

# Set or update baselines weekly
python -m time_series_performance baseline --all --days 7 --min-samples 3

# Check for regressions daily
python -m time_series_performance regression --days 1 --notify

# Generate weekly report
python -m time_series_performance report --days 30 --format html --output weekly_report.html
```

## Related Documentation

For more information, see:

- [Benchmark Database Guide](../BENCHMARK_DATABASE_GUIDE.md)
- [Visualization Guide](VISUALIZATION_GUIDE.md)
- [Benchmark Visualization Guide](benchmark_visualization.md)
- [CI/CD Integration Guide](CI_CD_INTEGRATION_GUIDE.md)
- [Performance Dashboard Specification](../PERFORMANCE_DASHBOARD_SPECIFICATION.md)