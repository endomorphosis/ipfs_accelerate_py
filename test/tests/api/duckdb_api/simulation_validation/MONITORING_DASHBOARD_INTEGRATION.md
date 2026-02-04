# Monitoring Dashboard Integration

The Simulation Accuracy and Validation Framework now includes a comprehensive integration with the monitoring dashboard system. This integration enables real-time visualization, alerting, and monitoring of simulation validation results.

## Overview

The dashboard integration connects the Simulation Validation Framework with the centralized monitoring dashboard used across the distributed testing framework. This allows simulation validation metrics to be displayed alongside other system metrics, providing a comprehensive view of the system's performance and accuracy.

Key features include:
- Real-time monitoring of simulation accuracy metrics
- Creation of customized dashboards with multiple visualization panels
- Alert configuration for simulation drift detection
- Automated panel creation based on hardware/model combinations
- Integration with the existing visualization components

## Architecture

The dashboard integration is built on a client-server architecture:
1. The `ValidationVisualizerDBConnector` serves as the client, connecting to the dashboard API
2. The monitoring dashboard server provides visualization rendering and alerting
3. Communication happens over a secure REST API with authentication
4. Data flows from the simulation validation database to the dashboard in real-time

```
┌───────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
│ Simulation Validation │         │ Validation          │         │ Monitoring          │
│ Database              │────────▶│ Visualizer          │────────▶│ Dashboard           │
│ (DuckDB)              │         │ DB Connector        │         │ Server              │
└───────────────────────┘         └─────────────────────┘         └─────────────────────┘
                                           │                                ▲
                                           │                                │
                                           ▼                                │
                                  ┌─────────────────────┐                   │
                                  │ Local Visualization │                   │
                                  │ Components          │───────────────────┘
                                  └─────────────────────┘
```

## Installation and Dependencies

To use the monitoring dashboard integration, you need to install the required dependencies:

```bash
# Run the installation script
./install_dashboard_integration_deps.sh

# Or install dependencies manually
pip install duckdb pandas fastapi uvicorn pydantic plotly requests websocket-client matplotlib seaborn
```

The installation script will create a virtual environment with all required dependencies:

```bash
# Activate the virtual environment after installation
source venv/bin/activate
```

## Configuration

To enable dashboard integration, use the following constructor parameters:

```python
connector = ValidationVisualizerDBConnector(
    db_integration=db_integration,  # Optional, will create one if not provided
    visualizer=visualizer,          # Optional, will create one if not provided
    dashboard_integration=True,     # Enable dashboard integration
    dashboard_url="http://dashboard.example.com/api",  # Dashboard API URL
    dashboard_api_key="your_api_key_here"             # API key for authentication
)
```

You can also use the simplified constructor with just the dashboard information:

```python
connector = ValidationVisualizerDBConnector(
    dashboard_integration=True,
    dashboard_url="http://localhost:8080/dashboard",
    dashboard_api_key="your_api_key"
)
```

## Creating Dashboard Panels

You can create individual panels for specific metrics:

```python
# Create a MAPE comparison panel
result = connector.create_dashboard_panel_from_db(
    panel_type="mape_comparison",
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metric="throughput_items_per_second",
    dashboard_id="my_dashboard",
    panel_title="MAPE Comparison",
    refresh_interval=60,  # Refresh every 60 seconds
    width=6,              # Width in grid units
    height=4              # Height in grid units
)

# Check result
print(f"Panel created: {result['panel_id']}")
print(f"Panel title: {result['title']}")
```

Supported panel types include:
- `mape_comparison`: Comparison of MAPE across hardware and models
- `hardware_heatmap`: Heatmap visualization of metrics across hardware types
- `time_series`: Time series chart showing how metrics change over time
- `simulation_vs_hardware`: Scatter plot comparing simulation and hardware values
- `drift_detection`: Visualization of simulation drift over time
- `calibration_effectiveness`: Chart showing improvements from calibration

### Example: Creating Multiple Panel Types

```python
# Create a MAPE comparison panel
mape_panel = connector.create_dashboard_panel_from_db(
    panel_type="mape_comparison",
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metric="throughput_items_per_second",
    dashboard_id="performance_dashboard",
    panel_title="BERT GPU Throughput MAPE"
)

# Create a hardware heatmap panel
heatmap_panel = connector.create_dashboard_panel_from_db(
    panel_type="hardware_heatmap",
    model_type="bert-base-uncased",
    metric="average_latency_ms",
    dashboard_id="performance_dashboard",
    panel_title="BERT Latency Across Hardware"
)

# Create a time series panel
time_series_panel = connector.create_dashboard_panel_from_db(
    panel_type="time_series",
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metric="throughput_mape",
    dashboard_id="performance_dashboard",
    panel_title="BERT GPU MAPE Over Time"
)
```

## Creating Comprehensive Dashboards

For a complete dashboard with multiple panels, use:

```python
# Create a comprehensive dashboard
result = connector.create_comprehensive_monitoring_dashboard(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    dashboard_title="BERT GPU Monitoring Dashboard",
    dashboard_description="Comprehensive monitoring of BERT model on RTX 3080",
    refresh_interval=60,  # Default refresh interval for panels
    include_panels=[      # List of panel types to include
        "mape_comparison",
        "hardware_heatmap",
        "time_series",
        "simulation_vs_hardware", 
        "drift_detection",
        "calibration_effectiveness"
    ]
)

# Access the dashboard URL
dashboard_url = result["url"]
print(f"Dashboard created: {result['dashboard_id']}")
print(f"Dashboard URL: {dashboard_url}")
```

The comprehensive dashboard creates a visually integrated view with optimized layout for all the selected panels. Each panel is configured with appropriate dimensions and positioning for optimal viewing.

### Example: Creating Targeted Dashboards

```python
# Create a dashboard focused on hardware comparison
hardware_dashboard = connector.create_comprehensive_monitoring_dashboard(
    dashboard_title="Hardware Comparison Dashboard",
    include_panels=["hardware_heatmap", "simulation_vs_hardware"],
    refresh_interval=300  # 5 minutes
)

# Create a dashboard focused on drift detection
drift_dashboard = connector.create_comprehensive_monitoring_dashboard(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    dashboard_title="BERT GPU Drift Monitoring",
    include_panels=["drift_detection", "time_series", "calibration_effectiveness"],
    refresh_interval=600  # 10 minutes
)

# Create a dashboard for a specific model family
model_dashboard = connector.create_comprehensive_monitoring_dashboard(
    model_type="vit-base-patch16-224",
    dashboard_title="ViT Model Family Dashboard",
    include_panels=["mape_comparison", "hardware_heatmap"],
    refresh_interval=1800  # 30 minutes
)
```

## Setting Up Real-Time Monitoring

The dashboard integration also supports real-time monitoring with alerting capabilities:

```python
# Set up real-time monitoring
result = connector.set_up_real_time_monitoring(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metrics=["throughput_mape", "latency_mape", "memory_mape", "power_mape"],
    monitoring_interval=300,  # Check every 5 minutes
    alert_thresholds={
        "throughput_mape": 15.0,  # Alert if MAPE exceeds 15%
        "latency_mape": 15.0,
        "memory_mape": 20.0,
        "power_mape": 25.0
    },
    dashboard_id="my_dashboard"  # Add monitoring panels to existing dashboard
)

print(f"Monitoring job created: {result['monitoring_job_id']}")
print(f"Next check at: {result['next_check']}")
```

This creates time series panels for each monitored metric and configures alerts when thresholds are exceeded. The monitoring system will automatically check for new validation results at the specified interval and update the dashboard in real-time.

### Example: Advanced Monitoring Setup

```python
# First create a dashboard
dashboard_result = connector.create_comprehensive_monitoring_dashboard(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    dashboard_title="BERT GPU Real-Time Monitoring"
)

# Set up different monitoring configurations
# 1. Throughput monitoring with high priority (checked every 5 minutes)
throughput_monitoring = connector.set_up_real_time_monitoring(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metrics=["throughput_mape"],
    monitoring_interval=300,  # 5 minutes
    alert_thresholds={"throughput_mape": 10.0},  # Lower threshold for high priority
    dashboard_id=dashboard_result["dashboard_id"]
)

# 2. Memory and power monitoring with lower priority (checked every 15 minutes)
resource_monitoring = connector.set_up_real_time_monitoring(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metrics=["memory_mape", "power_mape"],
    monitoring_interval=900,  # 15 minutes
    alert_thresholds={"memory_mape": 20.0, "power_mape": 25.0},
    dashboard_id=dashboard_result["dashboard_id"]
)

# 3. Drift monitoring for long-term trends (checked hourly)
drift_monitoring = connector.set_up_real_time_monitoring(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metrics=["overall_accuracy_score"],
    monitoring_interval=3600,  # 1 hour
    alert_thresholds={"overall_accuracy_score": 15.0},
    dashboard_id=dashboard_result["dashboard_id"]
)

# Print next check times
print(f"Throughput monitoring next check: {throughput_monitoring['next_check']}")
print(f"Resource monitoring next check: {resource_monitoring['next_check']}")
print(f"Drift monitoring next check: {drift_monitoring['next_check']}")
```

## Command-Line Interface

The dashboard integration provides command-line tools for easy use. The main script is `run_monitoring_dashboard_integration.py`:

```bash
# Run in synchronization mode
python run_monitoring_dashboard_integration.py --mode=sync --output-dir=./visualizations

# Run in auto-sync mode with continuous updates
python run_monitoring_dashboard_integration.py --mode=auto-sync --interval=300

# Create a dashboard panel
python run_monitoring_dashboard_integration.py --mode=panel --dashboard-id=my_dashboard

# Create a comprehensive dashboard
python run_monitoring_dashboard_integration.py --mode=dashboard

# Run a complete flow from visualization to dashboard integration
python run_monitoring_dashboard_integration.py --mode=run-and-sync
```

For dependency-free testing and demonstration, you can use `demo_monitoring_dashboard.py`:

```bash
# Run all demos
python demo_monitoring_dashboard.py --mode=all

# Create dashboard panels only
python demo_monitoring_dashboard.py --mode=panel

# Create comprehensive dashboards only
python demo_monitoring_dashboard.py --mode=dashboard

# Set up monitoring only
python demo_monitoring_dashboard.py --mode=monitor
```

## Error Handling and Reliability

The dashboard integration includes robust error handling:

1. **Connection Management**: Automatic reconnection if the dashboard connection is lost
2. **Authentication**: Secure token-based authentication with automatic token refresh
3. **Fallback Behavior**: Local visualization generation if dashboard is unavailable
4. **Error Reporting**: Comprehensive logging of dashboard communication errors

### Example: Handling Connection Issues

```python
# Initialize connector with error handling
connector = ValidationVisualizerDBConnector(
    dashboard_integration=True,
    dashboard_url="http://dashboard.example.com/api",
    dashboard_api_key="your_api_key"
)

# Check connection status
if not connector.dashboard_connected:
    print("Not connected to dashboard. Using local visualization fallback.")
    # Generate local visualization instead
    local_viz_path = connector.create_comprehensive_dashboard_from_db(
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        output_path="./local_dashboard.html"
    )
    print(f"Local dashboard created at: {local_viz_path}")
else:
    # Use dashboard integration
    result = connector.create_comprehensive_monitoring_dashboard(
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        dashboard_title="BERT GPU Monitoring Dashboard"
    )
    print(f"Dashboard created at: {result['url']}")
```

## Performance Considerations

The dashboard integration is designed to be lightweight and efficient:

- Incremental updates to minimize data transfer
- Client-side data processing to reduce server load
- Selective rendering to optimize dashboard performance
- Configurable refresh intervals to balance timeliness and efficiency

### Performance Optimization Tips

1. **Batch Creation**: Create multiple panels in a single dashboard creation call rather than individually
2. **Appropriate Refresh Intervals**: Use longer refresh intervals for metrics that change slowly
3. **Targeted Monitoring**: Monitor only the most critical metrics in real-time
4. **Data Filtering**: Filter data by time range, hardware type, and model type to reduce data volume
5. **Appropriate Resolution**: Use daily or weekly aggregation for long-term trends, hourly for recent data

## Example Use Cases

### Real-Time Drift Monitoring

```python
# Set up a dedicated drift monitoring dashboard
connector.create_comprehensive_monitoring_dashboard(
    dashboard_title="Simulation Drift Monitoring Dashboard",
    include_panels=["drift_detection", "time_series"],
    refresh_interval=300  # 5 minutes
)
```

### Model-Specific Accuracy Dashboard

```python
# Create a dashboard for a specific model across hardware types
connector.create_comprehensive_monitoring_dashboard(
    model_type="bert-base-uncased",
    dashboard_title="BERT Model Accuracy Dashboard",
    include_panels=["hardware_heatmap", "mape_comparison"]
)
```

### Hardware Comparison Dashboard

```python
# Create a dashboard comparing simulation accuracy across hardware
connector.create_comprehensive_monitoring_dashboard(
    dashboard_title="Hardware Simulation Accuracy Comparison",
    include_panels=["hardware_heatmap", "simulation_vs_hardware"]
)
```

### Calibration Effectiveness Monitoring

```python
# Create a dashboard to track calibration effectiveness
connector.create_comprehensive_monitoring_dashboard(
    dashboard_title="Simulation Calibration Effectiveness",
    include_panels=["calibration_effectiveness", "time_series"],
    refresh_interval=3600  # 1 hour
)
```

### Comprehensive Model Family Monitoring

```python
# Create a dashboard for each model family
model_families = ["bert", "vit", "whisper", "llama"]

for family in model_families:
    connector.create_comprehensive_monitoring_dashboard(
        model_type=f"{family}-base",
        dashboard_title=f"{family.upper()} Model Family Dashboard",
        include_panels=["mape_comparison", "hardware_heatmap", "drift_detection"],
        refresh_interval=1800  # 30 minutes
    )
```

## Integration with Testing Framework

The dashboard integration can be used in testing scripts to visualize test results:

```python
def test_simulation_vs_hardware():
    """Test simulation versus hardware accuracy."""
    # Set up connector with dashboard integration
    connector = ValidationVisualizerDBConnector(
        dashboard_integration=True,
        dashboard_url="http://localhost:8080/dashboard",
        dashboard_api_key="test_api_key"
    )
    
    # Run test
    test_results = run_simulation_hardware_comparison()
    
    # Create dashboard panel with results
    panel_result = connector.create_dashboard_panel_from_db(
        panel_type="simulation_vs_hardware",
        hardware_type="gpu_rtx3080",
        model_type="bert-base-uncased",
        metric="throughput_items_per_second",
        dashboard_id="test_results_dashboard",
        panel_title="Test: Simulation vs Hardware Comparison"
    )
    
    # Verify test results and dashboard creation
    assert test_results["accuracy"] > 0.9, "Simulation accuracy below threshold"
    assert panel_result["status"] == "success", "Failed to create dashboard panel"
    
    return panel_result["panel_id"]
```

## Integration with CI/CD Pipelines

The dashboard integration can be incorporated into CI/CD pipelines to automatically create dashboards for each build:

```bash
#!/bin/bash
# CI/CD script for dashboard integration

# Run tests and collect results
python run_comprehensive_tests.py --output-dir=./test_results

# Create a dashboard for the test results
python run_monitoring_dashboard_integration.py \
    --mode=dashboard \
    --dashboard-title="Build $(BUILD_NUMBER) Test Results" \
    --include-panels=mape_comparison,time_series,drift_detection \
    --input-dir=./test_results \
    --dashboard-url="${DASHBOARD_URL}" \
    --api-key="${DASHBOARD_API_KEY}"

# Set up monitoring for the new build
python run_monitoring_dashboard_integration.py \
    --mode=monitor \
    --dashboard-id="build_$(BUILD_NUMBER)" \
    --metrics=throughput_mape,latency_mape \
    --interval=300 \
    --alert-thresholds="throughput_mape=10.0,latency_mape=12.0" \
    --dashboard-url="${DASHBOARD_URL}" \
    --api-key="${DASHBOARD_API_KEY}"
```

## Future Enhancements

Planned enhancements for the dashboard integration include:

1. **Customizable Layouts**: User-defined dashboard layouts with drag-and-drop functionality
2. **Prediction Integration**: Integration with the Predictive Performance System
3. **Cross-Dashboard Linking**: Navigation between related dashboards
4. **Export Capabilities**: Exporting dashboard data and visualizations in various formats
5. **User Annotations**: Adding notes and annotations to dashboard panels
6. **Interactive Filtering**: Dynamic filtering of dashboard data

## Known Limitations

1. The dashboard integration requires a network connection to the dashboard server
2. Large datasets may cause slower refresh rates on the dashboard
3. Custom visualizations require additional configuration

## Troubleshooting

If you encounter issues with the dashboard integration:

1. Check the connection status using `connector.dashboard_connected`
2. Verify the dashboard URL and API key are correct
3. Ensure the required dependencies are installed
4. Check the logs for detailed error messages
5. Try using local visualization generation as a fallback

### Common Issues and Solutions

#### Connection Failures

```
Failed to connect to dashboard: Connection refused
```

**Solution**: Verify that the dashboard server is running and accessible at the provided URL.

#### Authentication Errors

```
Failed to connect to dashboard: 401 - Unauthorized
```

**Solution**: Check that the API key is correct and hasn't expired.

#### Dependency Issues

```
ImportError: No module named 'plotly'
```

**Solution**: Install the required dependencies:

```bash
./install_dashboard_integration_deps.sh
```

#### Data Retrieval Errors

```
Error creating dashboard panel: No validation results found in database
```

**Solution**: Verify that there is data in the database for the specified hardware type and model.

## Additional Resources

- [MONITORING_DASHBOARD_INTEGRATION_SUMMARY.md](../MONITORING_DASHBOARD_INTEGRATION_SUMMARY.md): Summary of the dashboard integration
- [DASHBOARD_INTEGRATION_COMPLETION.md](../DASHBOARD_INTEGRATION_COMPLETION.md): Completion report for the dashboard integration
- [SIMULATION_ACCURACY_VALIDATION_IMPLEMENTATION.md](../SIMULATION_ACCURACY_VALIDATION_IMPLEMENTATION.md): Implementation details for the Simulation Accuracy and Validation Framework
- [demo_monitoring_dashboard.py](../demo_monitoring_dashboard.py): Demo script for the dashboard integration
- [test_basic_dashboard_integration.py](../test_basic_dashboard_integration.py): Basic test script for the dashboard integration
- [install_dashboard_integration_deps.sh](../install_dashboard_integration_deps.sh): Script to install dependencies