# Qualcomm Power Metrics Analysis Guide (March 2025)

## Overview

This guide provides detailed instructions for using and analyzing the enhanced power metrics capabilities in the IPFS Accelerate Python framework. The March 2025 update significantly improves power monitoring for Qualcomm AI Engine deployments, with special focus on mobile and edge applications.

## Key Features of Enhanced Power Metrics

- **Model-Specific Power Profiling**: Tailored power monitoring based on model type
- **Battery Impact Analysis**: Practical metrics for mobile deployment planning
- **Thermal Management Insights**: Detection and analysis of thermal constraints
- **Energy Efficiency Metrics**: Performance normalized to energy consumption
- **DuckDB Integration**: Advanced query capabilities for metrics analysis

## Using Power Metrics in Your Code

### Basic Power Monitoring

```python
from test_ipfs_accelerate import QualcommTestHandler

# Initialize handler
handler = QualcommTestHandler()

# Run inference with power monitoring
result = handler.run_inference(
    model_path="path/to/model",
    input_data=my_input_data,
    monitor_metrics=True  # Enable power monitoring
)

# Access power metrics
power_metrics = result["metrics"]
print(f"Average power: {power_metrics['average_power_mw']} mW")
print(f"Peak power: {power_metrics['peak_power_mw']} mW")
print(f"Temperature: {power_metrics['temperature_celsius']}°C")
```

### Model-Type Aware Power Monitoring

```python
from test_ipfs_accelerate import QualcommTestHandler

# Initialize handler
handler = QualcommTestHandler()

# Run inference with model type specification for more accurate profiling
result = handler.run_inference(
    model_path="path/to/model",
    input_data=my_input_data,
    monitor_metrics=True,
    model_type="vision"  # Specify model type: vision, text, audio, llm
)

# Access enhanced metrics
metrics = result["metrics"]
print(f"Energy efficiency: {metrics['energy_efficiency_items_per_joule']} items/joule")
print(f"Battery impact: {metrics['battery_impact_percent_per_hour']}% per hour")
print(f"Thermal throttling detected: {metrics['thermal_throttling_detected']}")
```

### Using the Power Metrics API

The QualcommTestHandler includes a comprehensive API for power metrics:

```python
# Start monitoring with model-specific profiling
metrics_data = handler._start_metrics_monitoring(model_type="text")

# Perform inference or operations
# ...

# Stop monitoring and get comprehensive metrics
metrics = handler._stop_metrics_monitoring(metrics_data)

# Access all metrics
print(f"Power consumption: {metrics['power_consumption_mw']} mW")
print(f"Energy consumption: {metrics['energy_consumption_mj']} mJ")
print(f"Temperature: {metrics['temperature_celsius']}°C")
print(f"Energy efficiency: {metrics['energy_efficiency_items_per_joule']} items/joule")
print(f"Battery impact: {metrics['battery_impact_percent_per_hour']}% per hour")
```

## Running Tests with Power Metrics

### Command-Line Testing

```bash
# Enable Qualcomm testing with power metrics
export TEST_QUALCOMM=1

# Set database path for storing results
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# Run full test suite
python test/test_ipfs_accelerate.py

# Run tests for specific models
python test/test_qualcomm_integration.py --models bert-base-uncased,t5-small
```

### Mock Mode for Development

When actual Qualcomm hardware is unavailable, you can use mock mode for development and testing:

```bash
# Enable mock mode
export QUALCOMM_MOCK=1
export TEST_QUALCOMM=1

# Run tests with simulated power metrics
python test/test_ipfs_accelerate.py
```

The mock mode provides realistic simulated metrics based on the specified model type.

## Analyzing Power Metrics Data

### Basic Queries

```bash
# Query basic power consumption by model type
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
SELECT 
    model_type, 
    AVG(power_consumption_mw) as avg_power, 
    AVG(temperature_celsius) as avg_temp,
    COUNT(*) as count
FROM 
    power_metrics 
WHERE 
    hardware_type='qualcomm' 
GROUP BY 
    model_type 
ORDER BY 
    avg_power DESC
" --format table
```

### Energy Efficiency Analysis

```bash
# Compare energy efficiency across model types
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
SELECT 
    model_type, 
    AVG(energy_efficiency_items_per_joule) as efficiency,
    AVG(battery_impact_percent_per_hour) as battery_impact,
    COUNT(*) as count
FROM 
    power_metrics 
WHERE 
    hardware_type='qualcomm' 
GROUP BY 
    model_type 
ORDER BY 
    efficiency DESC
" --format chart --output efficiency_chart.png
```

### Thermal Analysis

```bash
# Analyze thermal throttling patterns
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
SELECT 
    model_type, 
    COUNT(*) as tests,
    SUM(CASE WHEN thermal_throttling_detected=true THEN 1 ELSE 0 END) as throttled_tests,
    ROUND(SUM(CASE WHEN thermal_throttling_detected=true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as throttling_pct,
    AVG(temperature_celsius) as avg_temp
FROM 
    power_metrics 
WHERE 
    hardware_type='qualcomm' 
GROUP BY 
    model_type 
ORDER BY 
    throttling_pct DESC
" --format table
```

### Battery Impact Assessment

```bash
# Estimate battery life for different model types
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
SELECT 
    model_type, 
    ROUND(AVG(battery_impact_percent_per_hour),1) as battery_pct_per_hour,
    ROUND(100/AVG(battery_impact_percent_per_hour),1) as est_hours_to_drain_battery
FROM 
    power_metrics 
WHERE 
    hardware_type='qualcomm' AND 
    battery_impact_percent_per_hour > 0 
GROUP BY 
    model_type 
ORDER BY 
    battery_pct_per_hour
" --format html --output battery_life.html
```

### Cross-Platform Comparison

```bash
# Compare power efficiency across hardware platforms
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
SELECT 
    hardware_type, 
    model_type,
    AVG(power_consumption_mw) as avg_power_mw,
    AVG(energy_efficiency_items_per_joule) as efficiency
FROM 
    power_metrics 
GROUP BY 
    hardware_type, model_type
ORDER BY 
    model_type, efficiency DESC
" --format table
```

## Generating Comprehensive Reports

### Power Efficiency Dashboard

```bash
# Generate comprehensive power efficiency report
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report power_efficiency --format html --output power_report.html

# Generate mobile-focused power analysis
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report mobile_power_efficiency --format html --output mobile_power_report.html
```

### Custom Visualizations

```bash
# Create battery impact visualization
python test/visualize_qualcomm_performance.py --report battery-impact --output battery_report.html

# Create thermal analysis visualization
python test/visualize_qualcomm_performance.py --report thermal --output thermal_report.html

# Create comprehensive dashboard with all metrics
python test/visualize_qualcomm_performance.py --report comprehensive --output dashboard.html
```

## Interpreting Power Metrics Results

### Energy Efficiency Interpretation

Energy efficiency (items/joule) provides a standardized metric across model types:

- **High Efficiency (>50 items/joule)**: Excellent for mobile deployment
- **Medium Efficiency (10-50 items/joule)**: Suitable for most use cases
- **Low Efficiency (<10 items/joule)**: May require optimization for mobile

### Battery Impact Guidelines

Battery impact estimates help with deployment planning:

- **Low Impact (<5% per hour)**: Suitable for background or continuous use
- **Medium Impact (5-15% per hour)**: Good for regular but not continuous use
- **High Impact (>15% per hour)**: Best for occasional use or with power supply

### Thermal Management

Thermal data helps identify potential throttling issues:

- **Low Temperature (<40°C)**: No thermal concerns
- **Medium Temperature (40-60°C)**: Normal operating range but monitor for throttling
- **High Temperature (>60°C)**: Likely to cause thermal throttling

## Advanced Usage

### Custom Power Profiles

You can define custom power profiles for specific models:

```python
# Define custom power profiles
CUSTOM_POWER_PROFILES = {
    "my-custom-model": {
        "base": 480.0,        # Base power in mW
        "variance": 60.0,     # Random variance for realistic simulation
        "peak_factor": 1.25,  # Peak power relative to base
        "idle_factor": 0.4    # Idle power relative to base
    }
}

# Use custom profiles in the handler
handler = QualcommTestHandler(custom_power_profiles=CUSTOM_POWER_PROFILES)
```

### Continuous Monitoring

For long-running operations, you can implement continuous monitoring:

```python
import time
from test_ipfs_accelerate import QualcommTestHandler

handler = QualcommTestHandler()
metrics_data = handler._start_metrics_monitoring(model_type="llm")

# Run operation with periodic metrics collection
metrics_snapshots = []
for i in range(10):
    # Do some work...
    time.sleep(1)
    
    # Take a snapshot without stopping monitoring
    if hasattr(handler, "_get_current_metrics"):
        snapshot = handler._get_current_metrics(metrics_data)
        metrics_snapshots.append(snapshot)
        print(f"Temperature at {i}s: {snapshot.get('temperature_celsius')}°C")

# Finally stop monitoring
final_metrics = handler._stop_metrics_monitoring(metrics_data)
```

## Best Practices for Power-Efficient Deployment

1. **Match Model Type to Hardware Capabilities**:
   - Vision models perform well on Qualcomm AI Engine with DSP optimizations
   - Text embeddings are extremely efficient in most configurations
   - Large LLMs may require careful optimization or scaling down

2. **Monitor Thermal Behavior**:
   - Always check thermal_throttling_detected in metrics
   - For mobile devices, implement cooling periods between intensive operations
   - Consider thermally-aware scheduling for long-running applications

3. **Optimize for Battery Life**:
   - Batch inference requests when possible to amortize power spikes
   - Use energy_efficiency_items_per_joule to compare model variants
   - Consider quantized models (INT8) for better power efficiency

4. **Database Storage Best Practices**:
   - Set BENCHMARK_DB_PATH environment variable consistently
   - Use timestamp-based queries for time-series analysis
   - Create periodic efficiency reports to track optimization progress

## Troubleshooting

### Common Issues and Solutions

1. **Missing Power Metrics**:
   - Check that monitor_metrics=True is set in run_inference
   - Verify that TEST_QUALCOMM environment variable is set
   - Confirm that the DuckDB database is properly initialized

2. **Unrealistic Power Values**:
   - For real hardware: check power monitoring API connections
   - For mock mode: model_type may be incorrectly detected

3. **Database Errors**:
   - Ensure database schema is up to date with latest version
   - Check that all required columns exist in power_metrics table

4. **Thermal Throttling Detection Issues**:
   - Verify temperature reporting works correctly
   - Check thermal_throttling_detected field in metrics
   - Consider device-specific thermal threshold adjustments

## Reference

### Complete Power Metrics Fields

| Field Name | Type | Description |
|------------|------|-------------|
| power_consumption_mw | FLOAT | Current power draw in milliwatts |
| energy_consumption_mj | FLOAT | Total energy used in millijoules |
| temperature_celsius | FLOAT | Device temperature during inference |
| monitoring_duration_ms | FLOAT | Duration of monitoring period |
| average_power_mw | FLOAT | Average power consumption |
| peak_power_mw | FLOAT | Maximum power observed |
| idle_power_mw | FLOAT | Baseline power draw |
| model_type | VARCHAR | Model category (vision, text, audio, llm) |
| energy_efficiency_items_per_joule | FLOAT | Performance per energy unit |
| thermal_throttling_detected | BOOLEAN | Throttling detection flag |
| battery_impact_percent_per_hour | FLOAT | Battery drain estimation |
| throughput | FLOAT | Performance metric |
| throughput_units | VARCHAR | Units based on model type |
| device_name | VARCHAR | Name of Qualcomm device |
| sdk_type | VARCHAR | SDK type (QNN or QTI) |
| sdk_version | VARCHAR | SDK version identifier |

### Environment Variables

| Variable | Description |
|----------|-------------|
| TEST_QUALCOMM | Enable Qualcomm testing (set to 1) |
| QUALCOMM_MOCK | Enable mock mode (set to 1) |
| BENCHMARK_DB_PATH | Path to DuckDB database |
| DEPRECATE_JSON_OUTPUT | Use database instead of JSON (set to 1) |
| QUALCOMM_SDK_TYPE | Override SDK type detection (QNN or QTI) |
| QUALCOMM_THERMAL_THRESHOLD | Override thermal throttling threshold |
| QUALCOMM_POWER_MONITOR_INTERVAL | Set monitoring interval in ms |