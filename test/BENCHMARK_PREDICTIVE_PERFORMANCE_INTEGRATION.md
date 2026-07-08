# Benchmark to Predictive Performance Integration

This guide explains how to use the integration between the Benchmark API and the Predictive Performance API, which allows benchmark results to be automatically synchronized with the predictive performance system for more accurate hardware recommendations and performance predictions.

## Overview

The Benchmark to Predictive Performance Bridge automates the process of transferring performance measurements from benchmark runs into the predictive performance system. This integration:

1. Connects directly to the benchmark database to access performance results
2. Converts benchmark results to the format expected by the predictive performance system
3. Records measurements through the Predictive Performance API
4. Tracks synchronization status to prevent duplicate measurements
5. Generates reports on synchronization status and coverage

## Components

The integration consists of several components:

1. **Benchmark Predictive Performance Bridge**: Core component that handles synchronization
2. **Bridge Configuration**: Configuration management for the integration
3. **Bridge Service**: Service for continuous synchronization
4. **Run Script**: Convenience script for running the integration

## Prerequisites

Before using the integration, ensure you have:

1. A running Benchmark API with results in its DuckDB database
2. A running Predictive Performance API server
3. The Unified API Server running with the Predictive Performance API integrated

## Running the Integration

There are multiple ways to run the integration:

### 1. As a One-Time Synchronization

To run a single synchronization cycle:

```bash
# Synchronize recent benchmark results
python test/run_benchmark_predictive_bridge.py --sync

# Synchronize results for a specific model
python test/run_benchmark_predictive_bridge.py --sync --model bert-base-uncased

# Specify custom database and API URL
python test/run_benchmark_predictive_bridge.py --sync \
  --benchmark-db ./path/to/benchmark_db.duckdb \
  --api-url http://localhost:8080
```

### 2. As a Continuous Service

To run the integration as a continuous service:

```bash
# Start the service with default configuration
python test/run_benchmark_predictive_bridge.py --service

# Use custom configuration and logging
python test/run_benchmark_predictive_bridge.py --service \
  --config ./bridge_config.json \
  --log-file ./bridge_service.log
```

### 3. Generate a Synchronization Report

To generate a report on the synchronization status:

```bash
# Generate a report
python test/run_benchmark_predictive_bridge.py --report

# Save report to a file
python test/run_benchmark_predictive_bridge.py --report --output sync_report.json
```

### 4. Create Default Configuration

To create a default configuration file:

```bash
# Create default configuration
python test/run_benchmark_predictive_bridge.py --create-config

# Specify output path
python test/run_benchmark_predictive_bridge.py --create-config --config my_config.json
```

## Configuration

The integration can be configured through a JSON configuration file. The default configuration includes:

```json
{
  "benchmark_predictive_performance": {
    "enabled": true,
    "benchmark_db_path": "benchmark_db.duckdb",
    "predictive_api_url": "http://localhost:8080",
    "api_key": null,
    "sync_interval_minutes": 60,
    "auto_sync_enabled": false,
    "sync_limit": 100,
    "sync_days_lookback": 30,
    "high_priority_models": [
      "bert-base-uncased",
      "gpt2",
      "t5-base", 
      "vit-base-patch16-224"
    ],
    "report_output_dir": "reports"
  },
  "logging": {
    "level": "INFO",
    "file": null,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

Key configuration parameters:

- `benchmark_db_path`: Path to the benchmark DuckDB database
- `predictive_api_url`: URL of the Unified API Server (or direct Predictive Performance API URL)
- `api_key`: Optional API key for authenticated endpoints
- `sync_interval_minutes`: Interval between synchronization cycles when running as a service
- `auto_sync_enabled`: Whether to automatically synchronize on a schedule
- `sync_limit`: Maximum number of results to synchronize per cycle
- `high_priority_models`: List of models that should be synchronized first
- `report_output_dir`: Directory to save synchronization reports

## Direct API Usage

You can also use the bridge directly in your own code:

```python
from test.integration.benchmark_predictive_performance_bridge import BenchmarkPredictivePerformanceBridge

# Create bridge instance
bridge = BenchmarkPredictivePerformanceBridge(
    benchmark_db_path="./benchmark_db.duckdb",
    predictive_api_url="http://localhost:8080"
)

# Check connections
status = bridge.check_connections()
print(f"Connection status: {status}")

# Synchronize recent results
result = bridge.sync_recent_results(limit=50)
print(f"Synced {result['synced']} of {result['total']} results")

# Generate report
report = bridge.generate_report()
print(f"Synchronization coverage: {report['sync_percentage']}%")

# Close connections
bridge.close()
```

## Integration with CI/CD

The integration can be included in CI/CD pipelines to ensure benchmark results are automatically synchronized with the predictive performance system.

Example GitHub Actions workflow:

```yaml
name: Sync Benchmark Results

on:
  workflow_run:
    workflows: ["Run Benchmarks"]
    types:
      - completed

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Start API servers
        run: |
          python test/run_integrated_api_servers.py &
          sleep 10  # Wait for servers to start
      
      - name: Sync benchmark results
        run: |
          python test/run_benchmark_predictive_bridge.py --sync
      
      - name: Generate report
        run: |
          python test/run_benchmark_predictive_bridge.py --report --output sync_report.json
      
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: sync-report
          path: sync_report.json
```

## Troubleshooting

### Common Issues

1. **Benchmark database not found**: Ensure the benchmark database exists and is specified correctly with `--benchmark-db`
2. **API servers not running**: Start the API servers with `python test/run_integrated_api_servers.py`
3. **Synchronization fails**: Check the log for details on why synchronization failed
4. **No benchmark results found**: Ensure benchmark results exist in the database
5. **Outdated performance predictions**: Use `--sync --model <model_name>` to force synchronization for a specific model

### Diagnostic Commands

Check database schema:

```python
import duckdb

# Connect to benchmark database
conn = duckdb.connect("benchmark_db.duckdb")

# List tables
print(conn.execute("SHOW TABLES").fetchall())

# Get schema for performance_results table
print(conn.execute("DESCRIBE performance_results").fetchall())

# Check for recent results
print(conn.execute("SELECT COUNT(*) FROM performance_results").fetchone()[0])
```

Check API connections:

```bash
# Check Unified API Server
curl http://localhost:8080/health

# Check Predictive Performance API
curl http://localhost:8080/api/predictive-performance/health
```

## Next Steps

1. **Create trained ML models**: Use synchronized measurements to train ML models for performance prediction
2. **Develop visualization dashboard**: Create dashboards for monitoring prediction accuracy
3. **Expand hardware coverage**: Add support for new hardware platforms
4. **Optimize recommendations**: Improve the recommendation algorithm based on benchmark results
5. **Implement automated analysis**: Detect performance issues and suggest optimizations