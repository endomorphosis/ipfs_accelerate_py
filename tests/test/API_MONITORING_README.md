# API Monitoring System Documentation

## Overview

The API Monitoring System is a comprehensive solution for monitoring API performance, detecting anomalies, and visualizing trends. It integrates with the API Distributed Testing Framework to provide real-time insights into API behavior.

## Architecture

The monitoring system consists of several key components:

### Data Collection

- **Test Results**: Collected from API Distributed Testing Framework
- **Real-time Metrics**: Latency, throughput, reliability, cost efficiency
- **Historical Data**: Stored for trend analysis and comparison

### Anomaly Detection

Implemented in `api_anomaly_detection.py`, the anomaly detection system uses multiple algorithms to identify unusual patterns in API behavior:

1. **Z-Score Analysis**: Identifies values that deviate significantly from the mean
2. **Moving Average Deviation**: Detects deviations from recent trends
3. **IQR (Interquartile Range)**: Identifies outliers based on quartile distribution
4. **Pattern Detection**: Recognizes repeating patterns and anomalies within them
5. **Seasonality-Aware Detection**: Considers time-of-day and day-of-week patterns

### Supported Anomaly Types

The system can detect various types of anomalies:

- **Latency Spikes**: Sudden increases in response time
- **Throughput Drops**: Unexpected decreases in request processing capacity
- **Error Rate Increases**: Abnormal increases in error responses
- **Cost Anomalies**: Unusual changes in tokens per dollar or other cost metrics
- **Pattern Changes**: Shifts in expected usage patterns

### Severity Levels

Anomalies are classified into severity levels:

- **LOW**: Minor deviations that may be normal variations
- **MEDIUM**: Significant deviations requiring attention
- **HIGH**: Critical issues that require immediate action

### Visualization

The monitoring dashboard provides:

- **Real-time Metrics Display**: Current API performance metrics
- **Historical Trend Analysis**: API performance over time
- **Anomaly Highlighting**: Visual indicators for detected anomalies
- **Comparative Analysis**: Side-by-side comparison of API providers
- **Performance Rankings**: API provider rankings based on various metrics

## Integration

### With API Distributed Testing Framework

The monitoring system integrates directly with the API Distributed Testing Framework:

```python
# Initialize the coordinator with monitoring enabled
coordinator = APICoordinatorServer(
    enable_anomaly_detection=True,
    enable_dashboard=True,
    dashboard_port=8080
)

# Start the coordinator
coordinator.start()
```

### With External Monitoring Systems

The system can integrate with external monitoring tools:

- **Prometheus Integration**: Export metrics to Prometheus
- **Grafana Dashboards**: Visualize metrics in Grafana
- **Alerting Systems**: Send alerts to email, Slack, PagerDuty, etc.

## Usage

### Accessing the Dashboard

The monitoring dashboard is available at:

```
http://<coordinator-host>:<dashboard-port>/
```

Default port is 8080.

### API Usage

You can programmatically access monitoring data:

```python
# Get metrics for all API providers
metrics = coordinator.get_metrics()

# Get metrics for a specific provider
openai_metrics = coordinator.get_metrics(provider="openai")

# Get anomalies
anomalies = coordinator.get_anomalies(days=7)
```

### Notification Configuration

Configure notifications for anomalies:

```python
notification_config = {
    "email": {
        "enabled": True,
        "recipients": ["alerts@example.com"],
        "min_severity": "MEDIUM"
    },
    "slack": {
        "enabled": True,
        "webhook_url": "https://hooks.slack.com/services/...",
        "channel": "#api-alerts",
        "min_severity": "HIGH"
    }
}

coordinator = APICoordinatorServer(
    notification_config=notification_config
)
```

## Advanced Features

### Custom Anomaly Detection Rules

You can define custom anomaly detection rules:

```python
from api_anomaly_detection import AnomalyDetector, AnomalyRule

# Create custom rule
rule = AnomalyRule(
    name="High Latency Spike",
    metric="latency",
    condition=lambda x: x > 500,  # Latency > 500ms
    severity="HIGH"
)

# Initialize detector with custom rule
detector = AnomalyDetector(custom_rules=[rule])
```

### Performance Analysis Reports

Generate comprehensive performance reports:

```python
# Generate a performance report
report_path = coordinator.generate_performance_report()

# Generate a report for specific providers
report_path = coordinator.generate_performance_report(
    providers=["openai", "claude"],
    output_file="api_comparison.json"
)
```

### Simulated Monitoring

The end-to-end example includes simulated monitoring capabilities:

```bash
python run_end_to_end_api_distributed_test.py
```

This generates performance metrics and anomaly detection without requiring actual API calls.

## Demo

To run a quick demonstration of the monitoring system:

```bash
# Start the coordinator server with dashboard
python run_api_coordinator_server.py --host 0.0.0.0 --port 5555

# In another terminal, start the end-to-end simulation
python run_end_to_end_api_distributed_test.py --mode simulation
```

Then access the dashboard at http://localhost:8080.

## Best Practices

1. **Regular Testing**: Run API tests regularly to build baseline metrics
2. **Custom Thresholds**: Adjust anomaly detection thresholds for your specific needs
3. **Notification Tuning**: Start with higher severity thresholds and adjust based on alert volume
4. **Historical Data**: Maintain historical data for long-term trend analysis
5. **Multi-metric Analysis**: Look at correlations between different metrics

## Troubleshooting

### Common Issues

1. **Dashboard not loading**:
   - Check that the coordinator server is running
   - Verify dashboard port is accessible
   - Check browser console for errors

2. **Missing anomaly detections**:
   - Verify anomaly detection is enabled
   - Check sensitivity settings
   - Ensure sufficient baseline data exists

3. **False positive anomalies**:
   - Adjust anomaly detection thresholds
   - Add seasonality awareness
   - Create custom rules for your use case

## Roadmap

1. **Enhanced Visualization**: More interactive charts and graphs
2. **Machine Learning Models**: More sophisticated anomaly detection
3. **Predictive Health Scores**: Overall API health metrics
4. **Custom Dashboards**: User-configurable dashboard layouts
5. **Mobile Apps**: Mobile-friendly monitoring interface

## References

- [API Distributed Testing Framework Guide](API_DISTRIBUTED_TESTING_GUIDE.md)
- [Predictive Analytics Documentation](PREDICTIVE_ANALYTICS_README.md)
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)