# Real-Time Performance Metrics Dashboard for Dynamic Resource Management

## Overview

The Real-Time Performance Metrics Dashboard for Dynamic Resource Management (DRM) provides comprehensive visualization and monitoring of resource allocation, utilization, and scaling decisions in the Distributed Testing Framework. This dashboard enables real-time observation of system performance, early detection of issues, and insight into optimization opportunities.

## Features

- **Real-Time Resource Monitoring**
  - CPU, memory, and GPU utilization tracking
  - Worker count and task load visualization
  - Resource allocation efficiency metrics

- **Worker-Level Metrics**
  - Individual worker performance tracking
  - Multi-worker comparison visualizations
  - Resource utilization heatmaps

- **Performance Analytics**
  - Task throughput and allocation time metrics
  - Statistical regression detection with confidence scoring
  - Trend analysis and forecasting

- **Scaling Decision Visualization**
  - Timeline of scaling events (scale up/down)
  - Correlation between utilization and scaling decisions
  - Decision reasoning and impact analysis

- **Alerting and Notification**
  - Automatic alerts for performance regressions
  - Customizable alert thresholds
  - Multi-level severity classification

## Getting Started

### Prerequisites

The dashboard requires the following dependencies:

```bash
pip install dash dash-bootstrap-components plotly numpy pandas scipy
```

### Running the Dashboard

To launch the dashboard with default settings:

```bash
python run_drm_real_time_dashboard.py
```

With custom parameters:

```bash
python run_drm_real_time_dashboard.py --port 8085 --theme dark --update-interval 5 --retention 60 --browser
```

### Command Line Options

- `--port`: Port to bind to (default: 8085)
- `--host`: Host to bind to (default: localhost)
- `--db-path`: Path to DuckDB database for historical data (default: benchmark_db.duckdb)
- `--theme`: Dashboard theme, light or dark (default: dark)
- `--update-interval`: Update interval in seconds (default: 5)
- `--retention`: Data retention window in minutes (default: 60)
- `--debug`: Enable debug mode
- `--browser`: Automatically open dashboard in browser
- `--drm-url`: URL of DRM coordinator (optional, for connecting to live systems)
- `--api-key`: API key for DRM coordinator authentication (optional)

## Dashboard Sections

### System Overview

The overview section provides a high-level summary of the system state, including:
- Current worker count and trend
- Active task count and trend
- Average resource utilization
- Alert count and status

Visual graphs show resource utilization over time and worker/task load patterns.

### Worker Details

The worker details section allows monitoring individual worker performance:
- Worker selection for comparison
- Metric selection (CPU, memory, GPU, tasks)
- Time-series graphs of selected metrics
- Resource utilization heatmap across workers

### Performance Metrics

The performance metrics section tracks key performance indicators:
- Task throughput metrics
- Resource allocation time
- Resource efficiency metrics
- Statistical regression analysis

### Scaling Decisions

The scaling decisions section visualizes the history of scaling events:
- Timeline of scaling decisions
- Correlation with resource utilization
- Detailed scaling reasons and impact

### Alerts and Notifications

The alerts section displays system alerts and notifications:
- Filterable by severity level and source
- Detailed alert messages with timestamps
- Regression detection alerts

## Integration with CI/CD

The dashboard can be integrated with CI/CD pipelines for automated monitoring:

```yaml
# GitHub Actions example
jobs:
  monitor-drm-performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements_dashboard.txt
      - name: Run DRM monitoring
        run: |
          python run_drm_real_time_dashboard.py --db-path ci_benchmark.duckdb --drm-url ${{ secrets.DRM_URL }} --api-key ${{ secrets.DRM_API_KEY }} --update-interval 10
```

## Programmatic Usage

The dashboard can also be used programmatically in Python code:

```python
from duckdb_api.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager

# Initialize DRM
drm = DynamicResourceManager()

# Create dashboard
dashboard = DRMRealTimeDashboard(
    dynamic_resource_manager=drm,
    port=8085,
    update_interval=5,
    retention_window=60,
    theme="dark"
)

# Start dashboard in background
dashboard.start_in_background()

# ... Your application code here ...

# Stop dashboard when done
dashboard.stop()
```

## Architecture

The dashboard system consists of several components:

1. **Data Collection Layer**: Periodically retrieves metrics from the DRM
2. **Analytics Engine**: Performs statistical analysis and regression detection
3. **Visualization Components**: Renders interactive charts and graphs
4. **Alerting System**: Monitors for significant events and anomalies
5. **Web Interface**: Dash-based UI for interaction and display

## Performance Considerations

- The dashboard is designed to have minimal impact on the DRM system
- Data collection occurs at configurable intervals (default: 5 seconds)
- Older data is automatically pruned based on retention window
- The dashboard can be run on a separate machine from the DRM coordinator

## Future Enhancements

Planned future enhancements include:

- Resource cost optimization recommendations
- Machine learning-based anomaly detection
- Predictive scaling based on historical patterns
- Additional export formats for reports
- Custom dashboard layout configuration
- Integration with alerting systems (Email, Slack, etc.)

## Troubleshooting

### Common Issues

**Dashboard shows no data**
- Ensure the DRM API URL and key are correct
- Check network connectivity to the DRM coordinator
- Verify the update interval is appropriate

**High CPU usage**
- Increase the update interval to reduce polling frequency
- Decrease the retention window to reduce memory usage
- Run in light theme mode which requires less rendering resources

**Missing regression detection**
- Install optional dependencies: `pip install scipy pandas`
- Ensure enough data points are available (at least 10)

## Contributing

Contributions to the dashboard are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Add tests for new functionality
5. Submit a pull request

## License

The dashboard is provided under the same license as the Distributed Testing Framework.