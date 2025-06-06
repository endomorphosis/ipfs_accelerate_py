# Monitoring Dashboard Guide

The Monitoring Dashboard provides a comprehensive, real-time visualization and monitoring system for the IPFS Accelerate Python Framework's distributed testing infrastructure. Implemented ahead of schedule (March 2025 vs. originally planned for June 2025), this dashboard enables real-time monitoring of all components in the distributed testing framework.

## Overview

The Monitoring Dashboard integrates with all aspects of the distributed testing framework to provide a unified view of:

- Worker node status and metrics
- Task execution and performance
- System topology and resource allocation
- Hardware utilization across the distributed system
- Fault tolerance metrics and recovery actions
- Alerts and notifications for critical events
- Performance trends and historical analysis

## Key Features

- **Real-time Monitoring**: Track system status, workers, tasks, and resources in real-time
- **Interactive Dashboard**: Web-based UI with responsive design
- **WebSocket Communication**: Real-time updates via WebSockets
- **Metrics Collection**: Comprehensive metrics tracking and storage
- **Visualization**: Interactive charts for performance metrics and system status
- **Alert Management**: System for tracking and managing alerts
- **REST API**: Comprehensive API for programmatic access to dashboard data

## Getting Started

### Prerequisites

The monitoring dashboard depends on the following Python packages:

```
tornado>=6.2.0          # Web server framework
websockets>=10.3.0      # WebSocket client support
plotly>=5.10.0          # Interactive visualizations
pandas>=1.4.0           # Data manipulation
numpy>=1.20.0           # Numerical computing
```

### Installation

Install the required packages:

```bash
pip install tornado websockets plotly pandas numpy
```

### Running the Dashboard

You can run the monitoring dashboard using the provided script:

```bash
python run_monitoring_dashboard.py
```

By default, this will start the dashboard server at http://localhost:8080.

#### Running with DRM Visualization

To run the monitoring dashboard with DRM visualization integration:

```bash
python run_dashboard_with_drm_visualization.py
```

This will start the dashboard with DRM visualization capabilities and generate mock data for demonstration purposes. The DRM dashboard will be available at http://localhost:8080/drm-dashboard.

### Command-Line Options

The dashboard supports various command-line options:

```
--host HOST                Host to bind the server to (default: localhost)
--port PORT                Port to bind the server to (default: 8080)
--coordinator-url URL      URL of the coordinator server
--db-path PATH             Path to SQLite database file
--auto-open                Open dashboard in browser automatically
--debug                    Enable debug logging
--generate-sample-data     Generate sample data for demonstration
```

Example with custom options:

```bash
python run_monitoring_dashboard.py --host 0.0.0.0 --port 8085 --coordinator-url http://coordinator-server:8000 --auto-open --debug
```

Example with sample data for demonstration:

```bash
python run_monitoring_dashboard.py --generate-sample-data --auto-open
```

#### DRM Visualization Dashboard Options

The DRM visualization dashboard supports additional options:

```
--no-mock                  Disable mock data generation (use real DynamicResourceManager)
```

Example with real resource manager:

```bash
python run_dashboard_with_drm_visualization.py --host 0.0.0.0 --port 8085 --no-mock
```

## Dashboard Components

The dashboard consists of multiple pages, each focused on a specific aspect of the distributed testing system:

### Overview Dashboard

The main dashboard provides a high-level overview of the entire system, with key metrics and status indicators:

- System status summary
- Active worker count and distribution
- Task execution statistics
- Recent alerts and notifications
- Performance trends
- Hardware utilization overview

### Dynamic Resource Management Dashboard

The DRM dashboard provides comprehensive visualizations for resource management:

- Resource utilization heatmaps across workers
- Scaling history and decision tracking
- Resource allocation and distribution visualization
- Resource efficiency metrics and optimization insights
- Cloud resource usage and cost tracking (if available)
- Interactive dashboard with real-time updates
- Configurable auto-refresh functionality
- Tabbed interface for easy navigation between visualization types

### System Status

The system status page provides detailed information about the overall system:

- Coordinator status and metrics
- Component health indicators
- Resource pool utilization
- System metrics history (CPU, memory, network)
- Queue statistics and processing rates

### Workers View

The workers view provides detailed information about worker nodes:

- Active/idle/disconnected worker counts
- Per-worker performance metrics
- Task allocation distribution
- Hardware capabilities and utilization
- Worker uptime and reliability statistics

### Tasks View

The tasks view provides insight into task execution:

- Running/completed/failed task counts
- Task execution time distribution
- Task types and categories
- Task dependencies and relationships
- Execution history and trends

### Performance View

The performance view focuses on performance metrics:

- Throughput and latency trends
- Hardware-specific performance comparisons
- Task type performance analysis
- Resource utilization efficiency
- Performance optimization recommendations

### Alerts View

The alerts view provides a history of system alerts:

- Critical/warning/info alert counts
- Alert timeline and distribution
- Alert source distribution
- Related metrics and contexts
- Alert acknowledgment and resolution tracking

### Topology View

The topology view visualizes the system's structure:

- Worker node relationships
- Resource allocation visualization
- Network connectivity map
- Task distribution across nodes
- Hardware capability distribution

### Fault Tolerance View

The fault tolerance view focuses on system resilience:

- Failure pattern detection
- Recovery action effectiveness
- Hardware-specific failure rates
- Mean time to recovery metrics
- Fault tolerance visualization

## Interactive Features

The dashboard supports several interactive features:

- Real-time updates via WebSockets
- Interactive data exploration
- Customizable refresh rates
- Theme switching (light/dark)
- Data filtering and sorting
- Detailed tooltips and popups

## API Endpoints

The dashboard exposes several API endpoints for programmatic access:

- `/api/status` - Overall system status
- `/api/workers` - Worker information
- `/api/tasks` - Task execution data
- `/api/metrics` - System metrics
- `/api/topology` - System topology
- `/api/performance` - Performance metrics
- `/api/alerts` - Alert history
- `/api/fault-tolerance` - Fault tolerance metrics

## Integration with Other Components

The monitoring dashboard integrates with several other components:

### Dynamic Resource Manager

The dashboard integrates with the Dynamic Resource Manager through the DRM Visualization Integration module to provide comprehensive resource monitoring and analysis:

- Resource utilization tracking and visualization
- Scaling decision monitoring and history
- Resource allocation efficiency analysis
- Cloud resource usage tracking
- Real-time updates via WebSockets
- Dashboard route at `/drm-dashboard` for dedicated resource monitoring

### Result Aggregator

The dashboard connects to the Result Aggregator Service to access aggregated test results, performance metrics, and statistical analyses.

### Coordinator

The dashboard connects to the Coordinator to monitor task distribution, worker management, and system topology.

### Fault Tolerance System

The dashboard integrates with the Hardware-Aware Fault Tolerance System to visualize failure patterns, recovery actions, and system resilience.

### Database

The dashboard can connect to the benchmark database (DuckDB) for historical data and trend analysis.

## Customization

The dashboard is highly customizable through configuration:

```python
dashboard.configure({
    "theme": "dark",                    # Dashboard theme (light, dark)
    "auto_refresh": 30,                 # Auto-refresh interval in seconds (0 to disable)
    "enable_alerts": True,              # Enable alert generation
    "real_time_enabled": True,          # Enable real-time updates via WebSockets
    "update_interval": 5,               # Background update interval in seconds
    "alert_retention_days": 7,          # How long to keep alert history
    "metrics_retention_days": 30,       # How long to keep metrics history
    "max_workers_shown": 50,            # Maximum number of workers shown in dashboard
    "max_tasks_tracked": 500,           # Maximum number of tasks tracked
    "enable_task_detail_tracking": True # Track detailed task execution
})
```

## Browser Compatibility

The dashboard is designed to work with all modern browsers:

- Chrome/Chromium: Full support
- Firefox: Full support
- Edge: Full support
- Safari: Full support with some WebSocket limitations

## Advanced Usage

### Programmatic Access

You can access the dashboard programmatically:

```python
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard

# Create dashboard instance
dashboard = MonitoringDashboard(
    host="localhost",
    port=8082,
    coordinator_url="http://coordinator-server:8080",
    output_dir="./monitoring_dashboard"
)

# Configure dashboard
dashboard.configure({
    "theme": "dark",
    "auto_refresh": 30,
    "enable_alerts": True
})

# Start dashboard asynchronously
thread = dashboard.start_async()

# Access dashboard data
status_data = dashboard._get_status_data()
print(f"Active workers: {status_data['workers']['active']}")

# Add custom alert
dashboard._add_alert(
    level="warning",
    title="Custom Alert",
    message="This is a custom alert message",
    source="custom_script",
    metrics={"value": 42}
)

# Stop dashboard when done
dashboard.stop()
```

### Custom Visualizations

You can create custom visualizations for the dashboard:

```python
from duckdb_api.distributed_testing.dashboard.visualization import VisualizationEngine

# Create visualization engine
viz_engine = VisualizationEngine(output_dir="./visualizations")

# Create custom visualization
viz_path = viz_engine.create_visualization(
    "time_series",
    {
        "time_series": {
            "metric1": [(datetime(2025, 3, 1), 100), (datetime(2025, 3, 2), 110)],
            "metric2": [(datetime(2025, 3, 1), 90), (datetime(2025, 3, 2), 95)]
        },
        "metric": "custom_metric",
        "title": "Custom Visualization"
    }
)
```

## Troubleshooting

### Common Issues

- **Dashboard Not Starting**: Check that all dependencies are installed and port is not in use
- **No Real-Time Updates**: WebSocket connections may be blocked by firewalls or proxies
- **Missing Visualizations**: Check that matplotlib is properly installed
- **No Data Displayed**: Verify connections to coordinator and result aggregator

### Logging

The dashboard produces detailed logs that can help diagnose issues:

```bash
python run_monitoring_dashboard.py --debug
```

This will enable debug-level logging with detailed information about dashboard operations.

## Future Enhancements

Planned enhancements for future releases:

- 3D visualization for multi-dimensional metrics
- Machine learning-based anomaly detection
- Predictive failure analysis
- Mobile app integration with push notifications
- Enhanced drill-down capabilities for root cause analysis
- Additional export formats (PDF, CSV, etc.)
- Advanced filtering and search capabilities