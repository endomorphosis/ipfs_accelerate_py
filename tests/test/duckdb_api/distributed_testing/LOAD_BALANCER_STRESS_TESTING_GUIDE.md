# Load Balancer Stress Testing Guide

This guide describes how to use the Load Balancer Stress Testing Framework to evaluate the performance, scalability, and stability of the Adaptive Load Balancer component.

## Overview

The Load Balancer Stress Testing Framework provides tools for:

1. **Single Stress Tests**: Evaluate performance under specific workloads
2. **Benchmark Suites**: Compare performance across different configurations
3. **Load Spike Simulations**: Assess behavior during load spikes and recovery
4. **Scenario-Based Testing**: Run predefined test scenarios from configuration

## Prerequisites

- Python 3.8 or higher
- Required dependencies:
  - `matplotlib` (for visualization)
  - `pandas` (for data analysis)
  - `numpy` (for numerical computations)
  - `seaborn` (for advanced visualizations)

If visualization dependencies are not installed, the framework will still work but won't be able to generate visual reports.

## Basic Usage

### Single Stress Test

Run a single stress test with default parameters:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress stress
```

Customize parameters:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress stress \
  --workers 10 \
  --tests 50 \
  --duration 30 \
  --burst \
  --dynamic \
  --output results.json
```

Parameters:
- `--workers`: Number of simulated workers (default: 20)
- `--tests`: Number of tests to submit (default: 100)
- `--duration`: Test duration in seconds (default: 60)
- `--burst`: Enable burst mode (tests submitted in bursts)
- `--dynamic`: Enable dynamic worker population (workers join/leave)
- `--output`: Output file for results (JSON format)

### Benchmark Suite

Run a benchmark suite that tests various combinations of workers and tests:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress benchmark
```

For a more comprehensive benchmark (can take a long time):

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress benchmark --full
```

### Load Spike Simulation

Simulate load spikes with dynamic worker population:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress spike \
  --workers 20 \
  --tests 100 \
  --duration 60 \
  --output spike_results.json
```

### Scenario-Based Testing

List available test scenarios:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress --list-scenarios
```

Run a specific scenario:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress \
  --scenario normal_operation \
  --output scenario_results.json
```

## Advanced Configuration

### Configuration File

The stress testing framework uses a configuration file (`load_balancer_stress_config.json`) to define test configurations, worker profiles, test profiles, and test scenarios.

You can specify a custom configuration file:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress \
  --config my_custom_config.json \
  --list-scenarios
```

### Available Scenarios

The default configuration includes the following scenarios:

1. **normal_operation**: Normal operation with steady test submission
2. **high_priority_burst**: Bursts of high-priority tests
3. **resource_contention**: Tests requiring more resources than typically available
4. **worker_churn**: Workers constantly joining and leaving the pool
5. **spike_and_recovery**: Load spikes combined with dynamic worker population

### Custom Scenarios

You can create custom scenarios by defining them in the configuration file:

```json
{
  "scenario_configurations": {
    "my_custom_scenario": {
      "test_config": "medium",
      "worker_profile": "balanced",
      "test_profile": "balanced",
      "burst_mode": true,
      "dynamic_workers": true,
      "description": "My custom test scenario"
    }
  }
}
```

## Visualization

After running tests, you can generate visualizations:

```bash
python -m duckdb_api.distributed_testing.visualize_load_balancer_performance \
  results.json \
  --output-dir ./visualizations
```

The visualization tool generates:

1. Performance summary charts
2. Throughput and latency time series
3. Worker utilization heat maps
4. Scalability analysis charts
5. Resource efficiency visualizations
6. Load spike correlation analysis

## Test Metrics

The stress test framework collects and reports the following metrics:

### Performance Metrics
- Success Rate: Percentage of tests that were successfully scheduled
- Average Latency: Average scheduling latency in seconds
- Min/Max Latency: Minimum and maximum scheduling latencies
- Peak Throughput: Maximum tests scheduled per second
- Average Throughput: Average tests scheduled per second

### Worker Distribution Metrics
- Worker Assignment Std Dev: Standard deviation of assignments across workers
- Worker Assignment Range: Difference between most and least assigned worker
- Worker Utilization: Percentage of available workers utilized
- Average Scheduling Attempts: Average number of attempts to schedule tests

### State Metrics
- Final Worker Count: Number of workers at test completion
- Pending Tests: Number of tests still pending
- Assigned Tests: Number of tests successfully assigned

## Common Scenarios

### Testing Scalability

To evaluate how well the load balancer scales with increasing worker counts:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress benchmark
```

### Testing Fault Tolerance

To evaluate how well the load balancer handles worker churn:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress \
  --scenario worker_churn
```

### Testing Load Spike Recovery

To evaluate how well the load balancer recovers from load spikes:

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress \
  --scenario spike_and_recovery
```

## Results Analysis

The framework generates detailed JSON results that can be used for further analysis.

Example results file structure:

```json
{
  "scenario": "normal_operation",
  "configuration": {
    "workers": 20,
    "tests": 100,
    "duration": 30,
    ...
  },
  "metrics": {
    "success_rate": 98.0,
    "avg_latency": 0.25,
    ...
  },
  "time_series": {
    "throughput": [[timestamp1, value1], [timestamp2, value2], ...],
    "latency": [[timestamp1, value1], [timestamp2, value2], ...]
  }
}
```

You can use tools like pandas, matplotlib, or custom scripts to analyze these results.

## Troubleshooting

### High Requeue Rate

If you see many tests being requeued and failing to schedule:

1. Check the worker profile - ensure workers have sufficient resources
2. Check the test profile - ensure tests don't require excessive resources
3. Adjust the worker-to-test ratio - ensure enough workers for the test load

### Performance Bottlenecks

If throughput is lower than expected:

1. Check CPU utilization during test runs
2. Adjust test submission rate or batch size
3. Look for resource contention (memory, locks)

### Missing Visualizations

If visualizations aren't generated:

1. Install required dependencies: `pip install matplotlib pandas numpy seaborn`
2. Check for error messages in the output
3. Ensure output directory exists and is writable

## Extending the Framework

### Adding New Scenarios

1. Edit the configuration file and add new scenario definitions
2. Define appropriate test configurations, worker profiles, and test profiles
3. Run with `--list-scenarios` to verify your scenario is available

### Adding New Metrics

1. Modify the `TestMetrics` class in `test_load_balancer_stress.py`
2. Add new metric collection logic
3. Update the `get_summary()` method to include your new metrics
4. Update result reporting to display the new metrics

### Adding New Visualization Types

1. Modify the `visualize_load_balancer_performance.py` script
2. Add new visualization functions
3. Update the main function to call your new visualization functions

## Conclusion

The Load Balancer Stress Testing Framework provides a comprehensive set of tools for evaluating the performance, scalability, and stability of the Adaptive Load Balancer component. By using these tools, you can identify potential issues, optimize configurations, and ensure the system performs optimally under real-world conditions.