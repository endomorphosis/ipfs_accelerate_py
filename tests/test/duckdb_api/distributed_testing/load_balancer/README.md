# Adaptive Load Balancer for Distributed Testing Framework

The Adaptive Load Balancer is a key component of the Distributed Testing Framework, designed to intelligently distribute tests across multiple worker nodes based on their capabilities, current load, and historical performance.

## Key Features

- **Worker capability detection**: Automatically identifies and tracks hardware and software capabilities of worker nodes
- **Performance-based scheduling**: Learns from historical test execution and improves scheduling decisions
- **Multiple scheduling algorithms**: Provides different strategies optimized for various use cases
- **Resource management**: Tracks and reserves memory, accelerators, and other resources
- **Concurrency control**: Ensures tests with the same concurrency key don't run simultaneously
- **Dynamic rebalancing**: Periodically rebalances workload across workers
- **Fault tolerance**: Properly handles tests that can't be scheduled

## Core Components

- `models.py`: Data models for worker capabilities, load, performance, test requirements, and assignments
- `capability_detector.py`: Worker hardware and software detection
- `performance_tracker.py`: Tracks and analyzes test execution metrics
- `scheduling_algorithms.py`: Different scheduling strategies (RoundRobin, PerformanceBased, etc.)
- `service.py`: Main load balancer service

## Usage

### Basic Example

```python
from duckdb_api.distributed_testing.load_balancer import (
    LoadBalancerService,
    WorkerCapabilities,
    WorkerLoad,
    TestRequirements
)

# Create load balancer
load_balancer = LoadBalancerService()

# Register callback for test status changes
def status_callback(assignment):
    print(f"Test {assignment.test_id} status: {assignment.status}")
    
load_balancer.register_assignment_callback(status_callback)

# Start load balancer
load_balancer.start()

# Register workers
worker_capabilities = WorkerCapabilities(
    worker_id="worker1",
    hostname="host1",
    supported_backends=["cpu", "cuda"],
    available_accelerators={"cuda": 2},
    available_memory=16.0,
    cpu_cores=8,
    cpu_threads=16
)
load_balancer.register_worker("worker1", worker_capabilities)
load_balancer.update_worker_load("worker1", WorkerLoad(worker_id="worker1"))

# Submit tests
test_requirements = TestRequirements(
    test_id="test1",
    model_id="bert-base-uncased",
    model_family="transformer",
    test_type="performance",
    minimum_memory=4.0,
    required_backend="cuda",
    expected_duration=60.0,
    priority=1,
    required_accelerators={"cuda": 1}
)
load_balancer.submit_test(test_requirements)

# Sometime later, stop the load balancer
load_balancer.stop()
```

### Worker Simulation

Workers connect to the load balancer and repeatedly request assignments:

```python
def worker_loop(worker_id, load_balancer):
    # Register with load balancer
    capabilities = get_worker_capabilities()
    load_balancer.register_worker(worker_id, capabilities)
    
    # Initialize load
    load = WorkerLoad(worker_id=worker_id)
    load_balancer.update_worker_load(worker_id, load)
    
    while running:
        # Get next assignment
        assignment = load_balancer.get_next_assignment(worker_id)
        
        if assignment:
            # Update status to running
            load_balancer.update_assignment_status(assignment.test_id, "running")
            
            # Execute test
            result = execute_test(assignment.test_requirements)
            success = result["success"]
            
            # Update status to completed/failed
            status = "completed" if success else "failed"
            load_balancer.update_assignment_status(assignment.test_id, status, result)
            
        # Wait before next poll
        time.sleep(1)
```

## Advanced Configuration

### Custom Schedulers

The load balancer supports different scheduling algorithms for different test types:

```python
from duckdb_api.distributed_testing.load_balancer import (
    create_scheduler,
    LoadBalancerService
)

# Create load balancer
load_balancer = LoadBalancerService()

# Set default scheduler
default_scheduler = create_scheduler("adaptive")
load_balancer.default_scheduler = default_scheduler

# Set test-type specific schedulers
performance_scheduler = create_scheduler("performance_based")
load_balancer.set_scheduler_for_test_type("performance", performance_scheduler)

priority_scheduler = create_scheduler("priority_based")
load_balancer.set_scheduler_for_test_type("critical", priority_scheduler)
```

### Composite Schedulers

Composite schedulers combine multiple strategies with different weights:

```python
config = {
    "type": "composite",
    "algorithms": [
        {"type": "performance_based", "weight": 0.6},
        {"type": "priority_based", "weight": 0.3},
        {"type": "affinity_based", "weight": 0.1}
    ]
}
scheduler = create_scheduler(**config)
```

## Architecture

### Worker Capacity Management

The load balancer carefully tracks worker capacity to prevent oversubscription:

1. When tests are submitted, they specify resource requirements (memory, accelerators, etc.)
2. The scheduler finds workers with compatible capabilities (hardware/software)
3. Before assignment, the system checks if the worker has enough available capacity
4. Resources are reserved on the selected worker
5. When tests complete, resources are released

### Fault Tolerance

The system handles cases where tests can't be scheduled:

1. Tests without compatible workers are requeued with decreased priority
2. After a configurable number of requeue attempts, tests are marked as failed
3. Test failures due to scheduling issues include detailed error information

### Performance Optimization

The load balancer continuously optimizes test placement:

1. Test execution metrics are recorded in the performance tracker
2. Performance data influences future scheduling decisions
3. Periodic rebalancing moves tests from overloaded workers to less utilized ones
4. Adaptive scheduling learns patterns and improves over time

## Testing

Use the test scripts to validate the load balancer:

- `test_fixed_load_balancer.py`: Verifies basic functionality and capacity management
- `test_basic_load_balancer.py`: Full example with simulated workers and tests

## Recommendations

- Configure `max_requeue_attempts` based on your environment (default is 5)
- Use appropriate scheduling algorithms for your workload patterns
- Provide accurate test requirements to improve scheduling decisions
- Implement custom scheduling algorithms for specialized needs