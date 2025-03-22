# DRM External Monitoring Integration Guide

This document provides comprehensive guidance on integrating the Dynamic Resource Management (DRM) system with external monitoring tools like Prometheus and Grafana.

## Overview

The DRM External Monitoring Integration enables exporting DRM performance metrics and resource utilization data to industry-standard monitoring systems, providing:

1. Real-time metrics in Prometheus format for comprehensive system monitoring
2. Pre-configured Grafana dashboards for rich visualizations and alerting
3. Customizable integration with existing monitoring infrastructure
4. Automatic detection of performance regressions and resource constraints

## Features

- **Prometheus Metrics Export**:
  - Resource utilization (CPU, memory, GPU)
  - Worker metrics (per-worker utilization, task counts)
  - Task metrics (active, pending, completed)
  - Performance metrics (throughput, allocation time, efficiency)
  - Scaling decisions (with reasons and directions)
  - Alerts and notifications (with severity levels)

- **Grafana Dashboard Integration**:
  - Pre-configured dashboard with comprehensive metrics visualization
  - Worker-specific panels with filtering
  - Performance trend analysis
  - Resource utilization heatmaps
  - System overview statistics
  - Scaling decision timelines

- **Integration Features**:
  - Automatic metric collection from DRM dashboard
  - Support for real DRM coordinators or simulated data
  - Template variables for dynamic filtering
  - Alerting rule templates for common issues
  - Export/import capabilities for dashboard configurations

## Installation

### Prerequisites

The External Monitoring Integration requires the following dependencies:

```bash
# Install core dependencies
pip install -r requirements_dashboard.txt

# Key dependencies
pip install prometheus_client requests
```

### External Systems Requirements

- **Prometheus**: Version 2.35+ recommended for optimal compatibility
- **Grafana**: Version 9.0+ recommended for all dashboard features

## Usage

### Starting the Integration

To start the DRM External Monitoring Integration:

```bash
python run_drm_external_monitoring.py
```

This will:
1. Launch the DRM dashboard with a mock DRM instance (for testing)
2. Start the Prometheus metrics exporter on port 9100
3. Generate a Grafana dashboard configuration
4. Open the DRM dashboard in a browser

### Command-Line Options

```
usage: run_drm_external_monitoring.py [-h] [--metrics-port METRICS_PORT] [--prometheus-url PROMETHEUS_URL]
                                    [--grafana-url GRAFANA_URL] [--grafana-api-key GRAFANA_API_KEY]
                                    [--dashboard-port DASHBOARD_PORT] [--update-interval UPDATE_INTERVAL]
                                    [--theme {light,dark}] [--output-dir OUTPUT_DIR] [--export-only]
                                    [--save-guide] [--drm-url DRM_URL] [--api-key API_KEY] [--simulation]
                                    [--no-browser]

DRM External Monitoring Integration

options:
  -h, --help            show this help message and exit
  --metrics-port METRICS_PORT
                        Port to expose Prometheus metrics (default: 9100)
  --prometheus-url PROMETHEUS_URL
                        Prometheus server URL
  --grafana-url GRAFANA_URL
                        Grafana server URL
  --grafana-api-key GRAFANA_API_KEY
                        Grafana API key for dashboard upload
  --dashboard-port DASHBOARD_PORT
                        DRM dashboard port (default: 8085)
  --update-interval UPDATE_INTERVAL
                        Update interval in seconds (default: 5)
  --theme {light,dark}  Dashboard theme
  --output-dir OUTPUT_DIR
                        Directory to save dashboard files
  --export-only         Only export Grafana dashboard without starting services
  --save-guide          Save integration guide to file
  --drm-url DRM_URL     URL of DRM coordinator (optional, for connecting to live DRM)
  --api-key API_KEY     API key for DRM coordinator authentication
  --simulation          Use simulated DRM data (default if no DRM URL provided)
  --no-browser          Don't automatically open browser
```

### Examples

**Start with simulated data (default):**
```bash
python run_drm_external_monitoring.py
```

**Connect to real DRM coordinator:**
```bash
python run_drm_external_monitoring.py --drm-url http://drm-coordinator:8080 --api-key YOUR_API_KEY
```

**Export Grafana dashboard only:**
```bash
python run_drm_external_monitoring.py --export-only --output-dir ./grafana_dashboards
```

**Customize ports and update interval:**
```bash
python run_drm_external_monitoring.py --metrics-port 9200 --dashboard-port 8090 --update-interval 10
```

**Save integration guide:**
```bash
python run_drm_external_monitoring.py --save-guide --export-only
```

## Prometheus Integration

### Metric Endpoint

The DRM External Monitoring Integration exposes metrics in Prometheus format at:

```
http://localhost:9100/metrics
```

### Prometheus Configuration

Add the following to your Prometheus configuration to scrape DRM metrics:

```yaml
scrape_configs:
  - job_name: 'drm'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9100']
```

### Available Metrics

Key metrics available through Prometheus:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `drm_cpu_utilization_percent` | Gauge | CPU utilization percentage across all workers |
| `drm_memory_utilization_percent` | Gauge | Memory utilization percentage across all workers |
| `drm_gpu_utilization_percent` | Gauge | GPU utilization percentage across all workers |
| `drm_worker_count` | Gauge | Total number of workers |
| `drm_active_workers` | Gauge | Number of active workers |
| `drm_active_tasks` | Gauge | Number of active tasks |
| `drm_pending_tasks` | Gauge | Number of pending tasks |
| `drm_completed_tasks_total` | Counter | Total number of completed tasks |
| `drm_task_throughput_per_second` | Gauge | Task throughput in tasks per second |
| `drm_allocation_time_milliseconds` | Gauge | Resource allocation time in milliseconds |
| `drm_resource_efficiency_percent` | Gauge | Resource efficiency percentage |
| `drm_scaling_operations_total` | Counter | Total number of scaling operations (labels: direction, reason) |
| `drm_alerts_total` | Counter | Total number of alerts (labels: level, source) |
| `drm_worker_cpu_utilization_percent` | Gauge | Worker CPU utilization percentage (labels: worker_id) |
| `drm_worker_memory_utilization_percent` | Gauge | Worker memory utilization percentage (labels: worker_id) |
| `drm_worker_gpu_utilization_percent` | Gauge | Worker GPU utilization percentage (labels: worker_id) |
| `drm_worker_tasks` | Gauge | Number of tasks on worker (labels: worker_id) |

### PromQL Examples

Common Prometheus queries for DRM metrics:

```promql
# Average CPU utilization over the last 5 minutes
avg_over_time(drm_cpu_utilization_percent[5m])

# High utilization detection
drm_cpu_utilization_percent > 80

# Worker count vs. active tasks
drm_worker_count / drm_active_tasks

# Scaling operation rate per minute
rate(drm_scaling_operations_total[1m])

# Count of alerts by level
sum by (level) (drm_alerts_total)

# Worker-specific utilization (top 5 by CPU)
topk(5, drm_worker_cpu_utilization_percent)
```

## Grafana Integration

### Dashboard Import

The integration generates a comprehensive Grafana dashboard that you can import:

1. Open Grafana web interface
2. Go to Dashboards > Import
3. Upload the JSON file from `monitoring_output/drm_dashboard.json` or paste its contents
4. Select your Prometheus data source
5. Click Import

### Dashboard Sections

The DRM dashboard includes the following sections:

1. **System Overview**:
   - At-a-glance metrics for workers, tasks, and utilization
   - Status indicators for system health
   - Quick filtering options

2. **Resource Utilization**:
   - Time series graphs of CPU, memory, and GPU utilization
   - Utilization by resource type and worker
   - Anomaly detection with visual indicators

3. **Performance Metrics**:
   - Task throughput and allocation time
   - Resource efficiency tracking
   - Performance trend analysis

4. **Worker Details**:
   - Per-worker resource utilization
   - Task distribution across workers
   - Filterable by worker ID or resource type

5. **Scaling Operations**:
   - Timeline of scaling decisions
   - Reasons for scaling operations
   - Correlation with resource utilization

6. **Alerts and Notifications**:
   - Alert history by severity level
   - Alert rate tracking
   - Alert source distribution

### Dashboard Customization

The dashboard is fully customizable in Grafana:

- Add additional panels for custom metrics
- Modify visualization types and colors
- Add custom alerting rules
- Adjust time ranges and refresh intervals
- Create dashboard variations for different use cases

## Programmatic Usage

The integration can be used programmatically in your Python code:

```python
from duckdb_api.distributed_testing.dashboard.drm_external_monitoring_integration import ExternalMonitoringBridge
from duckdb_api.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
from duckdb_api.distributed_testing.testing.mock_drm import MockDynamicResourceManager

# Create DRM dashboard
drm = MockDynamicResourceManager()  # or your real DRM instance
dashboard = DRMRealTimeDashboard(
    dynamic_resource_manager=drm,
    port=8085,
    update_interval=5
)

# Create and start monitoring bridge
bridge = ExternalMonitoringBridge(
    drm_dashboard=dashboard,
    metrics_port=9100,
    export_grafana_dashboard=True
)

# Start bridge
bridge.start()

# Start dashboard in background
dashboard.start_in_background()

# When done
bridge.stop()
dashboard.stop()
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: DRM Monitoring Integration

on:
  schedule:
    - cron: '0 * * * *'  # Hourly
  workflow_dispatch:

jobs:
  monitor-drm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements_dashboard.txt
      - name: Export monitoring data
        run: |
          python run_drm_external_monitoring.py --drm-url ${{ secrets.DRM_URL }} --api-key ${{ secrets.DRM_API_KEY }} --export-only --save-guide
      - name: Upload dashboards
        uses: actions/upload-artifact@v3
        with:
          name: grafana-dashboards
          path: monitoring_output/*.json
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'python -m pip install -r requirements_dashboard.txt'
            }
        }
        stage('Monitor DRM') {
            steps {
                sh 'python run_drm_external_monitoring.py --drm-url ${DRM_URL} --api-key ${DRM_API_KEY} --export-only --save-guide'
            }
        }
        stage('Archive') {
            steps {
                archiveArtifacts artifacts: 'monitoring_output/*.json', fingerprint: true
            }
        }
    }
}
```

## Advanced Features

### Alert Rule Configuration

Example Prometheus alert rules for DRM metrics:

```yaml
groups:
- name: DRM_Alerts
  rules:
  - alert: HighCPUUtilization
    expr: drm_cpu_utilization_percent > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU utilization detected"
      description: "CPU utilization has been over 85% for 5 minutes"

  - alert: LowWorkerCount
    expr: drm_worker_count < 3 and drm_pending_tasks > 10
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Worker shortage detected"
      description: "Less than 3 workers with over 10 pending tasks"

  - alert: HighAllocationTime
    expr: drm_allocation_time_milliseconds > 500
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow resource allocation"
      description: "Resource allocation time exceeds 500ms"
```

### Custom Metric Collection

You can extend the metrics collected by adding custom collectors:

```python
from prometheus_client import Gauge

# Create your custom gauge
custom_metric = Gauge('drm_custom_metric', 'Description of custom metric')

# Update the metric
custom_metric.set(value)
```

### Grafana Alert Configuration

Once your dashboard is imported, you can configure Grafana alerts:

1. Open a panel in edit mode
2. Go to the "Alert" tab
3. Configure alert conditions, e.g., "CPU utilization is above 80% for 5 minutes"
4. Add notifications channels (email, Slack, etc.)
5. Save the alert

## Troubleshooting

### Common Issues

**Prometheus metrics not available:**
- Check that port 9100 is not already in use
- Verify the prometheus_client package is installed
- Check for firewall rules that might block the port

**Dashboard doesn't start:**
- Verify all dependencies are installed
- Check permissions for the output directory
- Look for errors in the console output

**No data in Grafana:**
- Verify Prometheus is scraping the metrics endpoint
- Check Prometheus data source configuration in Grafana
- Verify time range in dashboard is appropriate

### Logs

Log files are saved to the console output by default. For more detailed logging:

```bash
python run_drm_external_monitoring.py --save-guide > monitoring.log 2>&1
```

### Support

For issues or questions:
1. Check the documentation in `DOCUMENTATION_INDEX.md`
2. Review the `monitoring_output/monitoring_integration_guide.md` file
3. Open an issue in the project repository

## Future Enhancements

Planned enhancements for future versions:

1. **Enhanced Alerting**:
   - Machine learning-based anomaly detection
   - Predictive alerts based on trend analysis
   - Alert correlation with root cause analysis

2. **Extended Integrations**:
   - Support for additional monitoring systems (Datadog, Dynatrace)
   - Cloud provider monitoring integrations (AWS CloudWatch, GCP Monitoring)
   - APM tool integration for distributed tracing

3. **Advanced Visualizations**:
   - 3D resource utilization visualizations
   - Interactive topology views of worker relationships
   - Custom dashboard templates for different user roles

4. **Scalability Improvements**:
   - Metric aggregation for large deployments
   - Hierarchical monitoring for multi-cluster setups
   - Selective metric collection for reduced overhead

## References

- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)
- [DRM Documentation](DYNAMIC_RESOURCE_MANAGEMENT.md)
- [Dashboard Documentation](REAL_TIME_PERFORMANCE_METRICS_DASHBOARD.md)