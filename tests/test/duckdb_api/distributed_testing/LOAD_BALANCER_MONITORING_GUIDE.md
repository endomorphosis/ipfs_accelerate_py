# Adaptive Load Balancer Monitoring Guide

This guide covers the monitoring capabilities of the Adaptive Load Balancer, including the live monitoring dashboard and performance visualization tools.

## Overview

The Adaptive Load Balancer includes robust monitoring capabilities that enable real-time visualization of performance metrics, throughput, latency, worker utilization, and test distribution. These monitoring tools are essential for:

1. Verifying load balancer performance under various conditions
2. Identifying bottlenecks and performance issues
3. Monitoring worker health and utilization
4. Visualizing test distribution and scheduling efficiency
5. Analyzing historical performance trends

## Live Monitoring Dashboard

The live monitoring dashboard (`load_balancer_live_dashboard.py`) provides a terminal-based real-time visualization of the load balancer's performance.

### Key Features

- Real-time throughput and latency monitoring
- Worker utilization visualization with thermal state indication
- Test queue monitoring with priority distribution
- System resource monitoring
- Time-series visualization of performance metrics

### Dashboard Components

The dashboard consists of several components:

1. **Key Metrics Panel**: Displays essential performance metrics like throughput, latency, worker count, and test counts.
2. **Throughput Graph**: Shows throughput over time as an ASCII chart.
3. **Worker Utilization Panel**: Displays CPU, memory, and GPU utilization for active workers.
4. **Pending Tests Panel**: Shows the queue of pending tests with priority distribution.
5. **Footer**: Displays uptime and system information.

### Using the Dashboard

The dashboard can be used in three modes:

#### 1. Monitor Mode

Attaches to a running load balancer to observe its performance:

```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard monitor
```

This mode is useful for passive monitoring of a load balancer that's already running as part of a larger system.

#### 2. Stress Test Mode

Runs a stress test while monitoring performance:

```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard stress \
  --workers 20 --tests 100 --duration 60 --burst --dynamic
```

Parameters:
- `--workers`: Number of simulated workers
- `--tests`: Number of tests to submit
- `--duration`: Test duration in seconds
- `--burst`: Submit tests in bursts rather than evenly distributed
- `--dynamic`: Add/remove workers during the test

#### 3. Scenario Mode

Runs a predefined test scenario while monitoring performance:

```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard scenario worker_churn
```

Parameters:
- `scenario`: Name of the scenario to run (e.g., worker_churn, resource_contention)
- `--config`: Optional path to a custom configuration file

### Interpreting the Dashboard

#### Throughput Metrics

- **Current Throughput**: Tests processed per second at the current moment
- **Average Throughput**: Average tests processed per second over time
- **Peak Throughput**: Maximum throughput achieved during the session

#### Latency Metrics

- **Current Latency**: Scheduling latency (time from submission to assignment) at the current moment
- **Average Latency**: Average scheduling latency over time
- **Min/Max Latency**: Minimum and maximum scheduling latency observed

#### Worker Utilization

The worker utilization panel shows:
- **CPU Utilization**: Percentage of CPU resources in use
- **Memory Utilization**: Percentage of memory resources in use
- **GPU Utilization**: Percentage of GPU resources in use (if applicable)
- **Status**: Worker thermal state (Normal, Warming, Cooling)

#### Test Distribution

The pending tests panel shows:
- **Total Pending**: Total number of tests waiting to be scheduled
- **Priority Distribution**: Breakdown of pending tests by priority level

## Offline Visualization

For more detailed analysis, the `visualize_load_balancer_performance.py` tool can generate comprehensive visualizations from test result files:

```bash
python -m duckdb_api.distributed_testing.visualize_load_balancer_performance \
  results.json --output-dir ./visualizations
```

This tool generates several types of visualizations:

### Single Test Visualizations

For stress test results:
- Performance summary charts
- Throughput and latency time series
- Worker utilization dashboard
- Success rate analysis

### Benchmark Visualizations

For benchmark suite results:
- 3D scalability analysis
- Worker utilization heatmaps
- Latency scaling charts
- Worker efficiency analysis
- Comparative benchmark dashboard

### Load Spike Visualizations

For spike simulation results:
- Throughput during spike visualization
- Latency during spike visualization
- Throughput vs latency correlation
- Time-series heat maps

## Programmatic Monitoring

For programmatic access to monitoring data, you can use the `DashboardMetricsCollector` class:

```python
from duckdb_api.distributed_testing.load_balancer_live_dashboard import DashboardMetricsCollector
from duckdb_api.distributed_testing.load_balancer import LoadBalancerService

# Create load balancer
balancer = LoadBalancerService()
balancer.start()

# Create metrics collector
metrics = DashboardMetricsCollector()

# Register callback
def assignment_callback(assignment):
    metrics.record_completion(assignment)
    
balancer.register_assignment_callback(assignment_callback)

# Run your tests...

# Get metrics summary
summary = metrics.get_summary()
print(f"Success rate: {summary['success_rate']}%")
print(f"Average latency: {summary['avg_latency']}s")
print(f"Peak throughput: {summary['peak_throughput']} tests/s")
```

## Best Practices for Monitoring

1. **Regular Benchmarking**: Run the benchmark suite regularly to track performance over time.
2. **Scenario Testing**: Use scenario-based testing to simulate real-world conditions.
3. **Live Monitoring During Deployment**: Use live monitoring when deploying changes to detect issues early.
4. **Performance History Analysis**: Save and compare visualization results to identify trends.
5. **Alert Thresholds**: Establish baseline performance and set alert thresholds for deviations.

## Troubleshooting

### Dashboard Not Displaying Correctly

If the dashboard is not displaying correctly:

1. Ensure your terminal supports ANSI escape sequences
2. Try adjusting terminal width and height
3. Check log output for any errors

### Visualization Dependencies

If you encounter errors with visualization:

```
Error: Required visualization libraries not found.
Please install required packages with:
pip install matplotlib pandas numpy seaborn
```

Install the required dependencies:

```bash
pip install matplotlib pandas numpy seaborn
```

### Performance Issues

If you observe performance issues during monitoring:

1. Reduce refresh rate by modifying `REFRESH_RATE` constant
2. Use smaller worker and test counts for initial testing
3. Check system resource utilization (CPU, memory) during testing

## Conclusion

The Adaptive Load Balancer monitoring tools provide comprehensive visibility into load balancer performance in real-time. By using these tools regularly, you can ensure optimal scheduling efficiency, identify potential issues early, and validate performance improvements over time.