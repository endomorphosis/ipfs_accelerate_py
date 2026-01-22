# Database Performance Monitoring Guide

This guide provides detailed information on setting up and using the Database Performance Monitoring system within the Simulation Accuracy and Validation Framework.

## Overview

The Database Performance Monitoring system provides real-time tracking and visualization of key database metrics to ensure optimal performance of the DuckDB database backend. It includes comprehensive metrics collection, status assessment, alerting, and dashboard integration.

## Getting Started

### 1. Using the Command-Line Tool

The framework provides a comprehensive command-line tool for database performance monitoring and management:

```bash
# Show current database performance metrics
python run_database_performance_monitoring.py metrics

# Output metrics in different formats
python run_database_performance_monitoring.py metrics --format json
python run_database_performance_monitoring.py metrics --format markdown --output metrics.md
python run_database_performance_monitoring.py metrics --format html --output metrics.html

# Show database statistics
python run_database_performance_monitoring.py stats

# Optimize database performance
python run_database_performance_monitoring.py optimize --compare

# Create a database backup
python run_database_performance_monitoring.py backup --compress

# Restore from a backup
python run_database_performance_monitoring.py restore --backup /path/to/backup.bak

# Clean up old records
python run_database_performance_monitoring.py cleanup --days 90 --dry-run
python run_database_performance_monitoring.py cleanup --days 90
```

### 2. Dashboard Integration with Command-Line Tool

```bash
# Create a database performance dashboard
python run_database_performance_monitoring.py dashboard \
    --url "http://dashboard.example.com/api" \
    --api-key "your-api-key" \
    --title "Database Performance Dashboard" \
    --style detailed \
    --update-interval 3600

# Create a complete monitoring solution
python run_database_performance_monitoring.py solution \
    --url "http://dashboard.example.com/api" \
    --api-key "your-api-key" \
    --title "Complete Monitoring Solution" \
    --hardware cuda cpu \
    --models bert vit \
    --style compact \
    --refresh-interval 300

# Update an existing dashboard
python run_database_performance_monitoring.py update \
    --url "http://dashboard.example.com/api" \
    --api-key "your-api-key" \
    --dashboard-id "dashboard-id"

# Continuous monitoring with real-time updates
python run_database_performance_monitoring.py monitor \
    --url "http://dashboard.example.com/api" \
    --api-key "your-api-key" \
    --dashboard-id "dashboard-id" \
    --interval 300 \
    --duration 3600  # optional: run for 1 hour, otherwise runs until interrupted
```

### 3. Programmatic Usage

#### Initialize the Performance Optimizer

```python
from duckdb_api.simulation_validation.db_performance_optimizer import DBPerformanceOptimizer

# Initialize the optimizer with your database path
optimizer = DBPerformanceOptimizer(
    db_path="./benchmark_db.duckdb",
    enable_caching=True,
    cache_size=100,
    cache_ttl=300
)
```

#### Collect Performance Metrics

```python
# Get comprehensive performance metrics
metrics = optimizer.get_performance_metrics()

# Check overall database status
status = optimizer.get_overall_status()
print(f"Database status: {status}")

# Check specific metrics
query_time = metrics["query_time"]["value"]
storage_size = metrics["storage_size"]["value"]
print(f"Query time: {query_time} ms")
print(f"Storage size: {storage_size / (1024 * 1024):.2f} MB")
```

#### Optimize Database Performance

```python
# Create indexes for improved query performance
optimizer.create_indexes()

# Analyze tables to update statistics
optimizer.analyze_tables()

# Run comprehensive optimization
optimization_result = optimizer.optimize_database()
if optimization_result:
    print("Database optimization completed successfully")
```

#### Backup and Restore

```python
# Create a backup of the database
backup_path = optimizer.backup_database()
print(f"Database backed up to: {backup_path}")

# Restore from a backup if needed
restore_result = optimizer.restore_database(backup_path)
if restore_result:
    print("Database restored successfully")
```

#### Cleanup Old Records

```python
# Remove records older than 90 days
cleanup_result = optimizer.cleanup_old_records(older_than_days=90)
for table, stats in cleanup_result.items():
    print(f"Cleaned up {stats['deleted']} records from {table}")
```

## Dashboard Integration

### 1. Set Up Dashboard Connection

```python
from duckdb_api.simulation_validation.visualization.monitoring_dashboard_connector import MonitoringDashboardConnector

# Create dashboard connector
dashboard_connector = MonitoringDashboardConnector(
    dashboard_url="http://monitoring.example.com/api",
    dashboard_api_key="your_api_key",
    db_optimizer=optimizer
)
```

### 2. Create a Database Performance Dashboard

```python
# Create a dedicated performance dashboard
dashboard_result = dashboard_connector.create_database_performance_dashboard(
    dashboard_title="Database Performance Monitoring",
    metrics=["query_time", "storage_size", "index_efficiency", "vacuum_status", "compression_ratio"],
    refresh_interval=300,  # 5 minutes
    visualization_style="detailed",
    create_alerts=True,
    auto_update=True,
    update_interval=3600  # 1 hour
)

print(f"Dashboard created at: {dashboard_result['dashboard_url']}")
print(f"Created {dashboard_result['panels_created']} panels and {dashboard_result['alerts_created']} alerts")
```

### 3. Update Dashboard Metrics Manually

```python
# Update dashboard metrics on demand
update_result = dashboard_connector.update_database_performance_metrics(
    dashboard_id=dashboard_result["dashboard_id"],
    include_history=True,
    format_values=True
)

print(f"Updated {update_result['updated_panels']} panels")
print(f"Overall status: {update_result['overall_status']}")
```

### 4. Create a Complete Monitoring Solution

```python
# Create a comprehensive dashboard with both database and validation metrics
solution_result = dashboard_connector.create_complete_monitoring_solution(
    dashboard_title="Comprehensive Monitoring Dashboard",
    include_database_performance=True,
    include_validation_metrics=True,
    hardware_types=["cuda", "cpu"],
    model_types=["bert", "vit"],
    performance_metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
    database_metrics=["query_time", "storage_size", "index_efficiency"],
    visualization_style="compact",
    refresh_interval=300
)

print(f"Complete solution created at: {solution_result['dashboard_url']}")
```

### 5. Continuous Monitoring

```python
# Set up continuous monitoring and dashboard updates
import time

try:
    interval = 300  # seconds
    print(f"Starting continuous monitoring (update every {interval} seconds)")
    
    iteration = 1
    while True:
        print(f"\nIteration {iteration} - {datetime.datetime.now().isoformat()}")
        
        # Get current metrics
        metrics = optimizer.get_performance_metrics()
        overall_status = optimizer.get_overall_status()
        
        # Display current status
        print(f"Overall database status: {overall_status.upper()}")
        
        # Update dashboard
        result = dashboard_connector.update_database_performance_metrics(
            dashboard_id=dashboard_result["dashboard_id"]
        )
        
        print(f"Updated {result['updated_panels']} dashboard panels")
        
        # Sleep until next update
        print(f"Next update in {interval} seconds...")
        time.sleep(interval)
        iteration += 1
        
except KeyboardInterrupt:
    print("\nMonitoring interrupted by user")
```

## Output Formats

The database performance monitoring system supports multiple output formats:

### 1. Text Format (Default)

Basic text output for console viewing:

```
=== Database Performance Metrics ===
Overall Status: GOOD
Timestamp: 2025-03-15T14:30:15.123456

✅ Query Time: 145.2 ms
   ↓ 12.5% from previous measurement

✅ Storage Size: 3.75 MB
   ↑ 5.2% from previous measurement

✅ Index Efficiency: 95.0%
```

### 2. JSON Format

Structured JSON output for programmatic consumption:

```json
{
  "metrics": {
    "query_time": {
      "value": 145.2,
      "previous_value": 166.0,
      "change_pct": -12.5,
      "unit": "ms",
      "status": "good",
      "history": [180, 175, 166, 145.2],
      "timestamp": "2025-03-15T14:30:15.123456"
    },
    "storage_size": {
      "value": 3932160,
      "previous_value": 3738624,
      "change_pct": 5.2,
      "unit": "bytes",
      "status": "good",
      "timestamp": "2025-03-15T14:30:15.123456"
    }
  },
  "overall_status": "good",
  "timestamp": "2025-03-15T14:30:15.123456"
}
```

### 3. Markdown Format

Formatted markdown for documentation:

```markdown
# Database Performance Metrics

**Overall Status:** GOOD

**Timestamp:** 2025-03-15T14:30:15.123456

| Metric | Value | Status | Change |
|--------|-------|--------|--------|
| Query Time | 145.2 ms | ✅ Good | ↓ 12.5% |
| Storage Size | 3.75 MB | ✅ Good | ↑ 5.2% |
| Index Efficiency | 95.0% | ✅ Good | ↑ 2.1% |
```

### 4. HTML Format

Rich HTML output for web display:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Database Performance Metrics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .status-good { color: green; }
        .status-warning { color: orange; }
        .status-error { color: red; }
        .metric-table { border-collapse: collapse; width: 100%; }
        /* Additional styles */
    </style>
</head>
<body>
    <h1>Database Performance Metrics</h1>
    <p><strong>Overall Status:</strong> <span class="status-good">GOOD</span></p>
    <p><strong>Timestamp:</strong> 2025-03-15T14:30:15.123456</p>
    
    <table class="metric-table">
        <!-- Table contents -->
    </table>
    
    <h2>Detailed Metrics</h2>
    <!-- Detailed metrics sections -->
</body>
</html>
```

## Available Metrics

The system collects and monitors the following database performance metrics:

| Metric | Description | Unit | Good Values | Warning Threshold | Error Threshold |
|--------|-------------|------|------------|-------------------|-----------------|
| query_time | Time to execute standard queries | ms | < 500 ms | 500 ms | 1000 ms |
| storage_size | Database file size | bytes | Depends on data volume | 500 MB | 1 GB |
| index_efficiency | Effectiveness of database indexes | % | > 70% | 70% | 50% |
| vacuum_status | Database fragmentation status | % | > 60% | 60% | 40% |
| compression_ratio | Data compression efficiency | ratio | > 2.5 | 2.5 | 1.5 |
| read_efficiency | Database read performance | records/s | > 200 | 200 | 100 |
| write_efficiency | Database write performance | records/s | > 150 | 150 | 75 |
| cache_performance | Query cache hit ratio | % | > 50% | 50% | 30% |

## Dashboard Visualization Styles

The system supports three visualization styles for the performance dashboard:

1. **Detailed** (default)
   - Large, comprehensive panels with extensive information
   - Two metrics per row for easy viewing
   - Includes trend charts and detailed status information
   - Best for in-depth analysis and troubleshooting

2. **Compact**
   - Condensed panels with essential information
   - Three metrics per row for space efficiency
   - Smaller trend charts and status indicators
   - Good for general monitoring on standard displays

3. **Overview**
   - Minimal panels with key information only
   - Four metrics per row for maximum space efficiency
   - Focus on current values and status indicators
   - Ideal for overview displays and dashboards with limited space

## Automated Monitoring

For hands-off monitoring, set up automatic updates and alerts:

```python
# Create dashboard with auto-updates
dashboard_result = dashboard_connector.create_database_performance_dashboard(
    dashboard_title="Automated Monitoring Dashboard",
    auto_update=True,
    update_interval=1800,  # 30 minutes
    create_alerts=True
)

# The dashboard will now automatically update every 30 minutes
# and send alerts when metrics exceed thresholds
```

Or using the command-line tool:

```bash
# Set up automated monitoring with alerts
python run_database_performance_monitoring.py dashboard \
    --url "http://dashboard.example.com/api" \
    --api-key "your-api-key" \
    --title "Automated Monitoring Dashboard" \
    --style detailed \
    --update-interval 1800
```

## Troubleshooting Common Issues

### High Query Time

If the `query_time` metric shows a warning or error status:

1. Check for complex queries running against large tables
2. Verify that appropriate indexes exist for frequently queried columns
3. Run `optimizer.create_indexes()` to create missing indexes
4. Run `optimizer.analyze_tables()` to update statistics

Command-line solution:
```bash
python run_database_performance_monitoring.py optimize
```

### Storage Growth

If the `storage_size` metric is increasing rapidly:

1. Run `optimizer.get_database_stats()` to identify large tables
2. Check if data retention policies should be implemented
3. Run `optimizer.cleanup_old_records()` to remove old data
4. Run `optimizer.optimize_database()` to reclaim space

Command-line solution:
```bash
python run_database_performance_monitoring.py stats
python run_database_performance_monitoring.py cleanup --days 90
python run_database_performance_monitoring.py optimize
```

### Low Index Efficiency

If the `index_efficiency` metric is poor:

1. Check if queries are using the correct columns for filtering
2. Verify that indexes exist on frequently filtered columns
3. Run `optimizer.create_indexes()` to create appropriate indexes

Command-line solution:
```bash
python run_database_performance_monitoring.py optimize --compare
```

### Low Vacuum Status

If the `vacuum_status` metric shows a warning or error status:

1. Run database optimization to reclaim space
2. Check for large tables that could be cleaned up

Command-line solution:
```bash
python run_database_performance_monitoring.py cleanup --days 90
python run_database_performance_monitoring.py optimize
```

## Automated Performance Optimization

The framework now includes an automated performance optimization system that can automatically detect issues and apply performance optimizations based on configurable thresholds.

### 1. Using the Automated Optimization Manager

```bash
# Check performance and get recommended optimizations
python run_database_performance_monitoring.py auto --action check

# Automatically detect and optimize performance issues
python run_database_performance_monitoring.py auto --action optimize --auto-apply

# Run a comprehensive optimization regardless of current metrics
python run_database_performance_monitoring.py auto --action comprehensive

# Analyze performance trends over time
python run_database_performance_monitoring.py auto --action trends --days 14

# Get optimization recommendations without applying them
python run_database_performance_monitoring.py auto --action recommendations

# Continuously monitor and automatically optimize (with auto-apply)
python run_database_performance_monitoring.py auto --action monitor --auto-apply --check-interval 1800
```

### 2. Configuration Options

The automated optimization system supports various configuration options:

```bash
# Specify a configuration file with custom thresholds and actions
python run_database_performance_monitoring.py auto --config /path/to/config.json

# Enable logging to a file
python run_database_performance_monitoring.py auto --action monitor --log-file /path/to/optimization.log

# Specify metrics to focus on
python run_database_performance_monitoring.py auto --action check --metrics query_time storage_size

# Configure the interval for continuous monitoring
python run_database_performance_monitoring.py auto --action monitor --check-interval 3600

# Set a specific duration for monitoring
python run_database_performance_monitoring.py auto --action monitor --monitor-time 7200

# Set number of days to retain data when cleaning up
python run_database_performance_monitoring.py auto --action optimize --auto-apply --retention-days 120
```

### 3. Configuration File Format

You can customize the automated optimization system using a JSON configuration file:

```json
{
  "thresholds": {
    "query_time": {
      "warning": 400.0,
      "error": 800.0
    },
    "storage_size": {
      "warning": 524288000,
      "error": 1073741824
    },
    "index_efficiency": {
      "warning": 75.0,
      "error": 60.0
    }
  },
  "actions": {
    "query_time": ["create_indexes", "analyze_tables"],
    "storage_size": ["vacuum_database", "cleanup_old_records"]
  },
  "check_interval": 1800,
  "auto_apply_actions": true,
  "retention_days": 90,
  "enable_scheduled_tasks": true,
  "history_days": 30
}
```

### 4. Scheduled Tasks

The automated optimization manager supports scheduled tasks for regular maintenance:

- **Daily tasks**: Create indexes and analyze tables
- **Weekly tasks**: Optimize and vacuum the database
- **Monthly tasks**: Clean up old records, optimize, and vacuum the database

These tasks are executed based on their defined intervals when continuous monitoring is enabled.

## Advanced Usage

### Scheduling Regular Monitoring

You can set up a cron job to regularly monitor database performance and send reports:

```bash
# Create a cron job to run every day at midnight
0 0 * * * python /path/to/run_database_performance_monitoring.py auto --action check --output /path/to/daily_report.json

# Create a cron job to run weekly optimization with auto-apply
0 0 * * 0 python /path/to/run_database_performance_monitoring.py auto --action comprehensive --auto-apply
```

### Database Maintenance Schedule

Here's a recommended maintenance schedule using the automated optimization system:

1. **Daily**: Run automated checks to detect issues
2. **Weekly**: Run comprehensive optimization
3. **Monthly**: Analyze trends and adjust thresholds
4. **Quarterly**: Review and update the automation configuration

Example weekly maintenance script:
```bash
#!/bin/bash
# Weekly database maintenance

# Path to the database
DB_PATH="/path/to/benchmark_db.duckdb"

# Run comprehensive optimization
python run_database_performance_monitoring.py auto --action comprehensive --auto-apply --db-path $DB_PATH

# Create a backup
TIMESTAMP=$(date +%Y%m%d)
python run_database_performance_monitoring.py backup --db-path $DB_PATH --output "/path/to/backups/backup_$TIMESTAMP.bak" --compress

# Generate a trend analysis report
python run_database_performance_monitoring.py auto --action trends --days 30 --db-path $DB_PATH --output "/path/to/reports/trend_analysis_$TIMESTAMP.json"
```

---

For further details, refer to the [Database Performance Implementation Summary](DATABASE_PERFORMANCE_IMPLEMENTATION_SUMMARY.md) document.

---

Guide updated on March 15, 2025