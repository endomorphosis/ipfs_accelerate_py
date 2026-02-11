# Hardware-Aware Scheduler for Distributed Testing Framework

This README provides an overview of the Hardware-Aware Scheduler implementation, which integrates the Hardware-Aware Workload Management system with the Load Balancer component in the Distributed Testing Framework.

## Components

The integration consists of the following key components:

1. **HardwareAwareScheduler**: A scheduling algorithm implementation that bridges the HardwareWorkloadManager and the LoadBalancerService, allowing for more intelligent scheduling based on hardware capabilities and workload characteristics.

2. **Load Balancer Integration Utilities**: Helper functions to simplify the creation and configuration of a hardware-aware load balancing system.

3. **Integration Example**: A comprehensive example demonstrating how to use the hardware-aware scheduler with various worker types and test requirements.

4. **Unit Tests**: Tests to verify the functionality of the hardware-aware scheduler and its integration with the load balancer.

5. **Documentation**: Detailed guide on how to use the integration components.

## Files

- `distributed_testing/hardware_aware_scheduler.py`: Core implementation of the hardware-aware scheduler
- `distributed_testing/load_balancer_integration.py`: Utilities for integrating with the load balancer
- `distributed_testing/examples/load_balancer_integration_example.py`: Example script demonstrating usage
- `distributed_testing/test_hardware_aware_scheduler.py`: Unit tests for the implementation
- `distributed_testing/HARDWARE_AWARE_SCHEDULER_GUIDE.md`: Comprehensive documentation

## Features

The Hardware-Aware Scheduler provides the following key features:

1. **Smart Hardware Matching**: Automatically matches test requirements to the most appropriate hardware based on detailed capability analysis.

2. **Thermal Management**: Tracks thermal states of hardware devices to avoid overheating and ensure consistent performance.

3. **Workload Learning**: Learns from execution history to improve future scheduling decisions, gradually optimizing hardware selection.

4. **Multi-Device Orchestration**: Supports splitting workloads across multiple devices or running them in parallel on multiple devices.

5. **Adaptive Efficiency Scoring**: Adjusts hardware efficiency scores based on current load, thermal state, and historical performance.

## Usage

### Basic Usage

```python
from distributed_testing.load_balancer_integration import create_hardware_aware_load_balancer

# Create load balancer with hardware-aware scheduling
load_balancer, workload_manager, scheduler = create_hardware_aware_load_balancer()

# Start the load balancer
load_balancer.start()

# Register workers
load_balancer.register_worker("worker1", worker_capabilities)

# Submit tests
load_balancer.submit_test(test_requirements)

# Shut down when done
load_balancer.stop()
workload_manager.stop()
```

### Advanced Usage with Composite Scheduler

```python
from distributed_testing.load_balancer_integration import create_hardware_aware_load_balancer

# Create load balancer with composite scheduler
load_balancer, workload_manager, scheduler = create_hardware_aware_load_balancer(
    use_composite=True,
    hardware_scheduler_weight=0.7  # 70% weight for hardware-aware decisions
)
```

## Tests

To run the unit tests:

```bash
python -m unittest distributed_testing.test_hardware_aware_scheduler
```

## Example

To run the example script:

```bash
python -m distributed_testing.examples.load_balancer_integration_example
```

The example demonstrates:
- Creating different types of workers (generic, GPU, TPU, browser, mobile)
- Generating various types of test requirements
- Matching tests to the most appropriate hardware
- Simulating execution and updating system state

## Further Documentation

For comprehensive documentation, please refer to [HARDWARE_AWARE_SCHEDULER_GUIDE.md](HARDWARE_AWARE_SCHEDULER_GUIDE.md), which covers:

- Architecture overview
- Component interactions
- Conversion between test requirements and workload profiles
- Worker capability to hardware profile mapping
- Scheduling process details
- Advanced features
- Best practices
- Troubleshooting
- Integration with existing schedulers
- Future enhancements