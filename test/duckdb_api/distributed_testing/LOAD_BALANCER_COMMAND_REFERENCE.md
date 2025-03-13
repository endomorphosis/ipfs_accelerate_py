# Adaptive Load Balancer Command Reference

This document provides a comprehensive reference for all command-line tools and arguments in the Adaptive Load Balancer system.

## Overview

The Adaptive Load Balancer includes several command-line tools for:
1. Stress testing
2. Benchmarking
3. Visualization
4. Live monitoring
5. Integration with other components

Each tool provides a variety of command-line options to customize behavior for different testing scenarios.

## Stress Testing Commands

### Basic Stress Test

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress stress \
  [--workers NUM] [--tests NUM] [--duration SECONDS] [--burst] [--dynamic] [--output FILE]
```

Arguments:
- `--workers`: Number of simulated workers (default: 20)
- `--tests`: Number of tests to submit (default: 100)
- `--duration`: Test duration in seconds (default: 60)
- `--burst`: Submit tests in bursts rather than evenly distributed
- `--dynamic`: Add/remove workers during the test
- `--output`: Save results to specified JSON file

Example:
```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress stress \
  --workers 50 --tests 200 --duration 120 --burst --dynamic --output large_stress_test.json
```

### Benchmark Suite

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress benchmark \
  [--full] [--output FILE]
```

Arguments:
- `--full`: Run the full benchmark suite (can take a long time)
- `--output`: Save results to specified JSON file

Example:
```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress benchmark \
  --full --output comprehensive_benchmark.json
```

### Load Spike Simulation

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress spike \
  [--workers NUM] [--tests NUM] [--duration SECONDS] [--output FILE]
```

Arguments:
- `--workers`: Initial number of workers (default: 20)
- `--tests`: Number of tests to submit (default: 100)
- `--duration`: Test duration in seconds (default: 60)
- `--output`: Save results to specified JSON file

Example:
```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress spike \
  --workers 30 --tests 500 --duration 180 --output spike_test_results.json
```

### Scenario-Based Testing

```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress \
  --scenario SCENARIO_NAME [--config CONFIG_FILE] [--output FILE]
```

Arguments:
- `--scenario`: Name of the scenario to run
- `--config`: Path to configuration file (optional)
- `--output`: Save results to specified JSON file

Example:
```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress \
  --scenario worker_churn --output worker_churn_results.json
```

List available scenarios:
```bash
python -m duckdb_api.distributed_testing.test_load_balancer_stress --list-scenarios
```

## Visualization Commands

### Generate Visualizations

```bash
python -m duckdb_api.distributed_testing.visualize_load_balancer_performance \
  INPUT_FILE [--output-dir DIR] [--type TYPE]
```

Arguments:
- `INPUT_FILE`: Path to the JSON results file
- `--output-dir`: Directory to save visualization files (default: ./visualizations)
- `--type`: Type of visualization (stress, benchmark, spike) - auto-detected if not specified

Example:
```bash
python -m duckdb_api.distributed_testing.visualize_load_balancer_performance \
  benchmark_results.json --output-dir ./benchmark_viz --type benchmark
```

## Live Monitoring Commands

### Monitor Mode

```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard monitor
```

This mode attaches to a running load balancer service for real-time monitoring.

### Stress Test with Live Monitoring

```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard stress \
  [--workers NUM] [--tests NUM] [--duration SECONDS] [--burst] [--dynamic]
```

Arguments:
- `--workers`: Number of simulated workers (default: 20)
- `--tests`: Number of tests to submit (default: 100)
- `--duration`: Test duration in seconds (default: 60)
- `--burst`: Submit tests in bursts rather than evenly distributed
- `--dynamic`: Add/remove workers during the test

Example:
```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard stress \
  --workers 10 --tests 50 --duration 30 --dynamic
```

### Scenario with Live Monitoring

```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard scenario SCENARIO_NAME \
  [--config CONFIG_FILE]
```

Arguments:
- `SCENARIO_NAME`: Name of the scenario to run
- `--config`: Path to configuration file (optional)

Example:
```bash
python -m duckdb_api.distributed_testing.load_balancer_live_dashboard scenario resource_contention
```

## Coordinator Integration Commands

### Run Coordinator with Load Balancer

```bash
python -m duckdb_api.distributed_testing.run_coordinator_with_load_balancer \
  [--host HOST] [--port PORT] [--db-path DB_PATH] [--scheduler SCHEDULER] \
  [--disable-load-balancer] [--log-level LEVEL]
```

Arguments:
- `--host`: Hostname for the coordinator server (default: localhost)
- `--port`: Port for the coordinator server (default: 8080)
- `--db-path`: Path to the DuckDB database file (default: ./benchmark_db.duckdb)
- `--scheduler`: Scheduler type to use (default: performance_based)
- `--disable-load-balancer`: Run without load balancer for comparison
- `--log-level`: Logging level (default: INFO)

Example:
```bash
python -m duckdb_api.distributed_testing.run_coordinator_with_load_balancer \
  --port 9090 --scheduler weighted_round_robin --log-level DEBUG
```

## Configuration File Format

The load balancer stress testing system uses JSON configuration files with the following structure:

```json
{
  "test_configurations": {
    "small": {
      "workers": 5,
      "tests": 20,
      "duration": 10,
      "description": "Small-scale test with few workers and tests"
    },
    "medium": {
      "workers": 20,
      "tests": 100,
      "duration": 30,
      "description": "Medium-scale test with moderate workers and tests"
    }
  },
  "worker_profiles": {
    "balanced": {
      "memory_distribution": [4, 8, 16, 32],
      "cuda_distribution": [0, 1, 2, 4],
      "description": "Balanced mix of CPU and GPU workers"
    },
    "cpu_heavy": {
      "memory_distribution": [8, 16, 32, 64],
      "cuda_distribution": [0, 0, 0, 1],
      "description": "Mostly CPU workers with high memory"
    }
  },
  "test_profiles": {
    "balanced": {
      "memory_requirements": [0.5, 1, 2, 4, 8],
      "cuda_requirements": [0, 0, 0, 1, 2],
      "priority_distribution": [1, 2, 2, 3, 3, 3, 4, 4, 5],
      "description": "Balanced mix of CPU and GPU tests"
    }
  },
  "scenario_configurations": {
    "normal_operation": {
      "test_config": "medium",
      "worker_profile": "balanced",
      "test_profile": "balanced",
      "burst_mode": false,
      "dynamic_workers": false,
      "description": "Normal operation with steady test submission"
    }
  }
}
```

## Output Format

The JSON output from stress tests and benchmarks follows this structure:

```json
{
  "configuration": {
    "workers": 20,
    "tests": 100,
    "duration": 60,
    "burst_mode": false,
    "dynamic_workers": true,
    "timestamp": "2025-03-15T10:30:15.123456"
  },
  "metrics": {
    "total_tests": 100,
    "success_rate": 98.0,
    "avg_latency": 0.25,
    "max_latency": 1.2,
    "min_latency": 0.05,
    "peak_throughput": 35,
    "avg_throughput": 15.5,
    "worker_assignment_stddev": 2.5,
    "worker_assignment_range": 8,
    "worker_utilization": 0.85,
    "avg_scheduling_attempts": 1.2,
    "final_worker_count": 22,
    "pending_tests": 0,
    "assigned_tests": 100
  },
  "time_series": {
    "throughput": [[timestamp1, value1], [timestamp2, value2], ...],
    "latency": [[timestamp1, value1], [timestamp2, value2], ...]
  }
}
```

## Environment Variables

The Adaptive Load Balancer supports the following environment variables:

- `LB_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `LB_MAX_WORKERS`: Maximum number of workers (for safety limits)
- `LB_DEFAULT_DURATION`: Default test duration in seconds
- `LB_CONFIG_PATH`: Default path to configuration file

Example:
```bash
LB_LOG_LEVEL=DEBUG LB_MAX_WORKERS=100 python -m duckdb_api.distributed_testing.test_load_balancer_stress benchmark
```

## Integration Examples

### Integrating with External Systems

```python
# Import needed components
from duckdb_api.distributed_testing.load_balancer import LoadBalancerService
from duckdb_api.distributed_testing.coordinator_load_balancer_integration import CoordinatorLoadBalancerIntegration

# Create and configure load balancer
load_balancer = LoadBalancerService()

# Create integration with external coordinator
integration = CoordinatorLoadBalancerIntegration(
    load_balancer=load_balancer,
    coordinator_url="http://localhost:8080",
    scheduler_type="performance_based"
)

# Start the integration
integration.start()

# Use the integration
integration.submit_task(task_data)

# When done
integration.stop()
```

### Using the API in Custom Scripts

```python
# Import components
from duckdb_api.distributed_testing.load_balancer import (
    LoadBalancerService,
    WorkerCapabilities,
    WorkerLoad,
    TestRequirements
)

# Create load balancer
balancer = LoadBalancerService()
balancer.start()

# Register workers
worker_id = "worker1"
capabilities = WorkerCapabilities(
    worker_id=worker_id,
    hardware_specs={
        "cpu": {"cores": 8, "frequency_mhz": 3000},
        "memory": {"total_gb": 16, "available_gb": 14.5},
        "gpu": {"device_count": 1, "cuda_available": True}
    },
    available_memory=14.5,
    cpu_cores=8,
    available_accelerators={"cuda": 1}
)
balancer.register_worker(worker_id, capabilities)

# Update worker load
load = WorkerLoad(
    worker_id=worker_id,
    cpu_utilization=25.0,
    memory_utilization=30.0,
    gpu_utilization=15.0
)
balancer.update_worker_load(worker_id, load)

# Submit tests
for i in range(10):
    test_id = f"test{i}"
    requirements = TestRequirements(
        test_id=test_id,
        model_id="bert-base-uncased",
        minimum_memory=4.0,
        priority=3,
        required_accelerators={"cuda": 1} if i % 2 == 0 else {}
    )
    balancer.submit_test(requirements)

# Process assignments
assignments = balancer.get_assignments()
for assignment in assignments:
    print(f"Test {assignment.test_id} assigned to {assignment.worker_id}")

# When done
balancer.stop()
```

## Conclusion

This command reference provides a comprehensive guide to all the available commands and options in the Adaptive Load Balancer system. For more detailed information on specific components, refer to the following documentation:

- [LOAD_BALANCER_IMPLEMENTATION_STATUS.md](LOAD_BALANCER_IMPLEMENTATION_STATUS.md)
- [LOAD_BALANCER_STRESS_TESTING_GUIDE.md](LOAD_BALANCER_STRESS_TESTING_GUIDE.md)
- [LOAD_BALANCER_MONITORING_GUIDE.md](LOAD_BALANCER_MONITORING_GUIDE.md)