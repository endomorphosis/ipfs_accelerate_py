# Time Series Performance Tracking System

**Date: March 7, 2025**  
**Status: Implemented**

## Overview

The Time Series Performance Tracking System is designed to track model performance metrics over time, detect performance regressions, visualize trends, and provide automated notifications when significant performance changes are detected. This system integrates with the DuckDB benchmark database and provides tools for analyzing historical performance data.

## Features

- **Versioned Test Results**: Track performance metrics across model versions
- **Regression Detection**: Automatically identify significant performance degradations
- **Trend Visualization**: Generate visualizations of performance metrics over time
- **Notification System**: Send alerts for critical performance issues

## Database Schema

The system extends the existing benchmark database with the following tables:

### Version History Table

Tracks model versions for correlating performance changes with code changes:

```sql
CREATE TABLE IF NOT EXISTS version_history (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version_tag VARCHAR,
    description VARCHAR,
    user_id VARCHAR,
    commit_hash VARCHAR,
    git_branch VARCHAR
)
```

### Performance Time Series Table

Stores time series performance data for trend analysis:

```sql
CREATE TABLE IF NOT EXISTS performance_time_series (
    id INTEGER PRIMARY KEY,
    version_id INTEGER,
    model_id INTEGER,
    hardware_id INTEGER,
    batch_size INTEGER,
    test_type VARCHAR,
    throughput FLOAT,
    latency FLOAT,
    memory_usage FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version_id) REFERENCES version_history(id),
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
)
```

### Regression Alerts Table

Tracks detected performance regressions:

```sql
CREATE TABLE IF NOT EXISTS regression_alerts (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_id INTEGER,
    hardware_id INTEGER,
    metric VARCHAR,
    previous_value FLOAT,
    current_value FLOAT,
    percent_change FLOAT,
    severity VARCHAR,
    status VARCHAR DEFAULT 'active',
    notification_sent BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
)
```

## Components

### TimeSeriesSchema

Handles database schema creation and management for time series data.

**Key Methods:**
- `create_schema_extensions()`: Creates required database tables if they don't exist
- `_get_connection()`: Gets a connection to the database with fallback support

### VersionManager

Manages version entries for tracking performance against specific code versions.

**Key Methods:**
- `create_version(version_tag, description, commit_hash, git_branch, user_id)`: Creates a new version entry
- `get_version(version_tag)`: Retrieves a version entry by tag
- `list_versions(limit)`: Lists recent version entries

### TimeSeriesManager

Manages the recording and retrieval of time series performance data.

**Key Methods:**
- `record_performance(model_name, hardware_type, batch_size, test_type, throughput, latency, memory_usage, version_tag)`: Records a performance metric
- `get_performance_history(model_name, hardware_type, batch_size, test_type, metric, days)`: Gets historical performance data

### RegressionDetector

Detects performance regressions by analyzing time series data.

**Key Methods:**
- `detect_regressions(model_name, hardware_type, metric, window_size, min_change)`: Identifies performance regressions
- `record_regressions(regressions)`: Records detected regressions in the database
- `list_active_regressions(severity, days)`: Lists active regression alerts
- `resolve_regression(regression_id, resolution_note)`: Marks a regression as resolved

**Regression Detection Algorithm:**
1. Collect historical performance data grouped by model, hardware, and batch size
2. Calculate baseline using a moving window of previous data points
3. Compare current performance against the baseline
4. Flag significant deviations based on configurable thresholds
5. Classify severity based on percent change

**Severity Classification:**
- **Critical**: > 25% degradation
- **High**: > 15% degradation
- **Medium**: > threshold (default 10%) degradation

### TrendVisualizer

Creates visualizations of performance trends over time.

**Key Methods:**
- `visualize_metric_trend(model_name, hardware_type, metric, batch_size, test_type, days, output_file, show_versions)`: Creates visualizations for a specific metric
- `create_regression_dashboard(days, limit, output_file)`: Creates a dashboard of recent regressions
- `create_comparative_dashboard(model_names, hardware_types, metric, days, output_file)`: Creates a dashboard comparing multiple models across hardware platforms

**Visualization Types:**
- Time series line charts with version markers
- Regression dashboards with severity indicators
- Comparative dashboards for cross-hardware and cross-model analysis

### NotificationSystem

Sends notifications for detected performance regressions.

**Key Methods:**
- `get_pending_notifications()`: Gets regression alerts that need notifications
- `mark_notification_sent(alert_id)`: Marks a regression alert as notified
- `create_github_issue(regression, repository)`: Creates a GitHub issue for a regression
- `send_email_notification(regression, recipients, smtp_settings)`: Sends email notifications
- `process_notifications(github_repo, email_recipients, smtp_settings)`: Processes all pending notifications

**Notification Strategy:**
- **Critical**: GitHub issue + email notification
- **High**: GitHub issue only
- **Medium**: Dashboard notification only

## Usage

### Command-Line Interface

The system provides a command-line interface for various operations:

```bash
# Create schema extensions
python test/time_series_performance_tracker.py create-schema --db-path ./benchmark_db.duckdb

# Create a version entry
python test/time_series_performance_tracker.py create-version --tag "v1.0.0" --description "Initial release" --commit "abc123" --branch "main"

# Record performance metric
python test/time_series_performance_tracker.py record --model "bert-base-uncased" --hardware "cuda" --batch-size 16 --throughput 98.5 --latency 10.2 --memory 1024 --version "v1.0.0"

# Detect performance regressions
python test/time_series_performance_tracker.py detect --model "bert-base-uncased" --hardware "cuda" --metric throughput --threshold 0.1 --window 5 --record

# Create trend visualization
python test/time_series_performance_tracker.py visualize --model "bert-base-uncased" --hardware "cuda" --metric throughput --days 30 --output trend.png

# Create regression dashboard
python test/time_series_performance_tracker.py dashboard --days 30 --limit 10 --output dashboard.png

# Create comparison dashboard
python test/time_series_performance_tracker.py compare --models "bert-base-uncased,t5-small,vit-base" --hardware "cuda,rocm,cpu" --metric throughput --output comparison.png

# Process notifications
python test/time_series_performance_tracker.py notify --github-repo "youorg/yourrepo"
```

### Programmatic API

The system can also be used programmatically:

```python
from test.time_series_performance_tracker import (
    TimeSeriesSchema, 
    VersionManager,
    TimeSeriesManager,
    RegressionDetector, 
    TrendVisualizer, 
    NotificationSystem
)

# Create schema if needed
schema = TimeSeriesSchema()
schema.create_schema_extensions()

# Create a version entry
version_manager = VersionManager()
version_id = version_manager.create_version(
    version_tag="v1.0.0",
    description="Initial release",
    commit_hash="abc123",
    git_branch="main"
)

# Record performance metrics
ts_manager = TimeSeriesManager()
ts_manager.record_performance(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    batch_size=16,
    test_type="inference",
    throughput=98.5,
    latency=10.2,
    memory_usage=1024,
    version_tag="v1.0.0"
)

# Detect regressions
detector = RegressionDetector(threshold=0.1)
regressions = detector.detect_regressions(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    metric="throughput",
    window_size=5
)

# Record detected regressions
detector.record_regressions(regressions)

# Visualize trends
visualizer = TrendVisualizer()
visualizer.visualize_metric_trend(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    metric="throughput",
    days=30,
    output_file="trend.png"
)

# Create comparative dashboard
visualizer.create_comparative_dashboard(
    model_names=["bert-base-uncased", "t5-small"],
    hardware_types=["cuda", "cpu"],
    metric="throughput",
    output_file="comparison.png"
)

# Process notifications
notifier = NotificationSystem()
notifier.process_notifications(
    github_repo="youorg/yourrepo",
    email_recipients=["team@example.com"]
)
```

## Integration with CI/CD Pipeline

The system integrates with the CI/CD pipeline for automated performance monitoring:

1. Performance tests run as part of CI/CD workflow
2. Results are stored in the time series database
3. Regression detection runs automatically after tests
4. Notifications are sent for detected regressions
5. Visualizations are generated and published

## Future Enhancements

1. **Statistical Analysis**: Add more sophisticated statistical methods for outlier detection
2. **Predictive Analytics**: Implement predictive models for forecasting performance trends
3. **Interactive Dashboard**: Create a web-based interactive dashboard for trend analysis
4. **Correlation Analysis**: Identify correlations between code changes and performance changes
5. **Automated Benchmarking**: Schedule periodic benchmarks for continuous monitoring

## Implementation Status

- ✅ Database Schema Extensions
- ✅ Version Management System
- ✅ Time Series Data Management
- ✅ Regression Detection Algorithm
- ✅ Trend Visualization System
- ✅ Comparative Dashboard Creation
- ✅ Notification System with GitHub and Email Integration
- 🔄 CI/CD Integration (In Progress)
- 🔄 Interactive Dashboard (Planned)
- ❓ Predictive Analytics (Future)

## References

- [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md)
- [PERFORMANCE_DASHBOARD_SPECIFICATION.md](PERFORMANCE_DASHBOARD_SPECIFICATION.md)
- [NEXT_STEPS.md](NEXT_STEPS.md)