# Hardware-Aware Scheduler Integration Guide

This guide explains how to integrate the Hardware-Aware Workload Management system with the Load Balancer component in the Distributed Testing Framework, using the new `HardwareAwareScheduler`.

## Overview

The Hardware-Aware Scheduler serves as a bridge between the Hardware-Aware Workload Management system and the Load Balancer component. It enables more intelligent scheduling decisions based on detailed hardware capabilities, workload characteristics, and thermal states.

![Architecture Diagram](https://via.placeholder.com/800x400?text=Hardware-Aware+Scheduler+Architecture)

Key benefits:
- **Smart hardware matching**: Tests are allocated to the most appropriate hardware for their specific needs
- **Thermal awareness**: Considers device thermal states for optimal long-term performance
- **Workload classification**: Automatically categorizes workloads for better hardware compatibility
- **Hardware learning**: Learns which hardware performs best for specific workload types
- **Multi-device orchestration**: Supports sharded and replicated execution across multiple devices

## Components

The integration involves the following components:

1. **Hardware Taxonomy**: Provides detailed classifications and relationships between hardware types
2. **Hardware Workload Manager**: Manages workload characterization and hardware matching
3. **Load Balancer Service**: Handles test scheduling and worker management
4. **Hardware-Aware Scheduler**: Implements the scheduling algorithm interface to bridge these systems

## Getting Started

### Prerequisites

Before implementing the Hardware-Aware Scheduler, ensure you have:

- A functioning Load Balancer component (from `duckdb_api.distributed_testing.load_balancer`)
- The Hardware-Aware Workload Management system (from `distributed_testing.hardware_workload_management`)
- The Hardware Taxonomy system (from `duckdb_api.distributed_testing.hardware_taxonomy`)

### Basic Implementation

Here's how to set up the Hardware-Aware Scheduler:

```python
from duckdb_api.distributed_testing.load_balancer.service import LoadBalancerService
from duckdb_api.distributed_testing.hardware_taxonomy import HardwareTaxonomy
from distributed_testing.hardware_workload_management import HardwareWorkloadManager
from distributed_testing.hardware_aware_scheduler import HardwareAwareScheduler

# Create components
hardware_taxonomy = HardwareTaxonomy()
workload_manager = HardwareWorkloadManager(hardware_taxonomy)
scheduler = HardwareAwareScheduler(workload_manager, hardware_taxonomy)

# Configure load balancer to use hardware-aware scheduling
load_balancer = LoadBalancerService()
load_balancer.default_scheduler = scheduler

# Start services
workload_manager.start()
load_balancer.start()
```

## Test Requirements to Workload Profiles

The Hardware-Aware Scheduler automatically converts test requirements from the Load Balancer into workload profiles for the Hardware Workload Manager.

### Key Mappings

| Test Requirement | Workload Profile |
|------------------|-----------------|
| `test_id` | `workload_id` |
| `test_type` | `workload_type` (converted to enum) |
| `model_id` | Stored in `custom_properties` |
| `minimum_memory` | `min_memory_bytes` (converted from GB) |
| `expected_duration` | `estimated_duration_seconds` |
| `priority` | `priority` (same scale) |
| `required_backend` | `required_backends` (converted to enum set) |
| Custom properties | Mapped to sharding and allocation strategy |

## Worker Capabilities to Hardware Profiles

The Hardware-Aware Scheduler also converts worker capabilities to hardware capability profiles for the hardware taxonomy.

For each worker, it:
1. Detects the hardware types available (CPU, GPU, TPU, NPU, etc.)
2. Creates appropriate profiles for each hardware type
3. Registers these profiles with the hardware taxonomy
4. Maintains a mapping between worker IDs and hardware IDs

## Scheduling Process

When the scheduler needs to select a worker for a test:

1. The test requirements are converted to a workload profile
2. The workload manager finds compatible hardware with efficiency scores
3. The scheduler filters by available workers and checks load capacity
4. Efficiency scores are adjusted based on current load and thermal state
5. The worker with the highest adjusted efficiency is selected

## Advanced Features

### Thermal Management

The scheduler tracks thermal states for workers to avoid overloading and ensure consistent performance:

- **Warming state**: Devices that have been idle need time to warm up to peak performance
- **Cooling state**: Devices under high load may need cooling time to avoid throttling
- **Performance levels**: Adjusted based on thermal state (0.0-1.0 scale)

### Workload Type Learning

The scheduler learns which workers perform best for specific workload types:

```python
# Update workload-worker preferences with experience
alpha = 0.3  # Learning rate 
current_preference = self.workload_worker_preferences.get(workload_type, {}).get(worker_id, 0.5)
updated_preference = (1 - alpha) * current_preference + alpha * efficiency
```

This enables better scheduling decisions over time as the system learns from experience.

## Multi-Device Orchestration

The scheduler supports three allocation strategies:

1. **Single**: Allocate workload to a single device (default)
2. **Sharded**: Split workload across multiple devices for parallel execution
3. **Replicated**: Run the same workload on multiple devices for redundancy

For sharded and replicated execution, the workload requirements need special properties:

```python
test_requirements.custom_properties = {
    "is_shardable": True,
    "min_shards": 2,
    "max_shards": 4,
    "allocation_strategy": "sharded"  # or "replicated"
}
```

## Example Implementation

See the included example script for a complete demonstration:
`distributed_testing/examples/load_balancer_integration_example.py`

The example:
1. Creates a variety of worker types (generic, GPU, TPU, browser, mobile)
2. Generates synthetic test requirements with various characteristics
3. Demonstrates the scheduling and execution processes
4. Shows how workloads are matched to appropriate hardware

## Best Practices

1. **Worker Registration**: Register workers with accurate and detailed capability information
2. **Test Requirements**: Include as much information as possible in test requirements to enable better matching
3. **Custom Properties**: Use custom properties to specify advanced features like sharding
4. **Load Updates**: Keep worker load information up-to-date for accurate scheduling decisions

## Troubleshooting

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| Tests not being scheduled | Ensure worker capabilities match test requirements |
| Inefficient hardware matching | Check workload profile conversion from test requirements |
| Performance degradation | Verify thermal state tracking and adjustment |
| Memory allocation errors | Ensure accurate memory reporting in worker capabilities |

## Integration with Existing Schedulers

The Hardware-Aware Scheduler can be combined with other scheduling algorithms through the `CompositeScheduler`:

```python
from duckdb_api.distributed_testing.load_balancer.scheduling_algorithms import (
    CompositeScheduler, PriorityBasedScheduler, RoundRobinScheduler
)

# Create composite scheduler with multiple algorithms
composite = CompositeScheduler([
    (hardware_aware_scheduler, 0.7),  # 70% weight for hardware-aware scheduling
    (PriorityBasedScheduler(), 0.2),  # 20% weight for priority-based scheduling
    (RoundRobinScheduler(), 0.1)      # 10% weight for round-robin scheduling
])

# Use the composite scheduler
load_balancer.default_scheduler = composite
```

## Visualization System

The Hardware-Aware Scheduler includes a comprehensive visualization system to help understand and analyze scheduling decisions, hardware efficiency, and system performance. The visualization capabilities are implemented in the `HardwareSchedulingVisualizer` class.

### Visualization Types

The visualization system supports several types of visualizations:

1. **Hardware Efficiency**: Visualize efficiency scores for different hardware profiles for a specific workload
2. **Workload Distribution**: Visualize the distribution of workloads across different workers
3. **Thermal States**: Visualize the thermal states (temperature, warming/cooling) of workers
4. **Resource Utilization**: Visualize CPU, memory, GPU, I/O, and network utilization across workers
5. **Execution Times**: Compare estimated and actual execution times for workloads

### Historical Data Tracking

The visualizer includes a history tracking system that records:

- Workload assignments with efficiency scores
- Thermal state changes over time
- Resource utilization changes over time
- Execution time comparisons (estimated vs. actual)

This historical data can be:
- Visualized as time-series charts
- Saved to JSON files for later analysis
- Used to generate comprehensive HTML reports

### Usage Example

```python
from distributed_testing.hardware_aware_visualization import create_visualizer

# Create visualizer
visualizer = create_visualizer(output_dir="visualization_results")

# Track workload assignments
visualizer.record_assignment(
    workload_id="workload_123", 
    worker_id="worker1", 
    efficiency_score=0.85,
    workload_type="VISION"
)

# Track thermal states
visualizer.record_thermal_state(
    worker_id="worker1", 
    temperature=0.7,
    warming_state=False,
    cooling_state=True
)

# Track resource utilization
visualizer.record_resource_utilization(
    worker_id="worker1", 
    utilization={
        "cpu_utilization": 75.0,
        "memory_utilization": 60.0,
        "gpu_utilization": 80.0,
        "io_utilization": 30.0,
        "network_utilization": 40.0
    }
)

# Track execution times
visualizer.record_execution_time(
    workload_id="workload_123", 
    estimated_time=60.0,
    actual_time=65.5,
    workload_type="VISION",
    worker_id="worker1"
)

# Visualize history data
visualizer.visualize_history(
    history_data=visualizer.history,
    filename_prefix="history_summary"
)

# Generate HTML report
visualizer.generate_summary_report(
    filename="scheduling_summary.html",
    include_visualizations=True
)
```

### Visualization Examples

For comprehensive examples of the visualization capabilities, see the example script:
`distributed_testing/examples/visualization_example.py`

This script demonstrates all the visualization types with synthetic data and real integration with the Hardware-Aware Scheduler.

To run the visualization examples:

```bash
python -m distributed_testing.examples.visualization_example
```

You can run specific examples:

```bash
python -m distributed_testing.examples.visualization_example --example efficiency
python -m distributed_testing.examples.visualization_example --example thermal
python -m distributed_testing.examples.visualization_example --example integrated
```

## Future Enhancements

Planned improvements to the Hardware-Aware Scheduler:

1. **Machine Learning-Based Scoring**: Use ML models to predict efficiency scores based on historical data
2. **Energy-Optimized Scheduling**: Consider energy efficiency in scheduling decisions
3. **Cost-Based Scheduling**: Factor in resource costs when selecting hardware
4. **Dynamic Resource Scaling**: Automatically scale resources based on load patterns
5. **Performance History Analysis**: More sophisticated use of execution history for predictions
6. **Advanced Visualization Dashboard**: Interactive web-based dashboard for real-time scheduling monitoring
7. **Predictive Timing Model**: Machine learning-based model for execution time prediction
8. **Automated Parameter Tuning**: Self-optimization of scheduling parameters based on performance data

## Conclusion

The Hardware-Aware Scheduler provides a powerful integration between the Hardware-Aware Workload Management system and the Load Balancer component. It enables more intelligent, efficient, and hardware-optimized test execution across heterogeneous environments.

By leveraging detailed hardware taxonomy and workload characterization, it ensures tests are run on the most appropriate hardware, improving performance, resource utilization, and energy efficiency.

The integrated visualization system enhances understanding of the scheduling decisions and system performance, providing valuable insights for system operators and developers.