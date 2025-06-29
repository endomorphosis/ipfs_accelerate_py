# Custom Scheduler Extensibility for Distributed Testing Framework

This document describes the custom scheduler extensibility system implemented for the distributed testing framework as part of Phase 8 (Integration and Extensibility).

## Overview

The custom scheduler extensibility system enables users to create, configure, and use custom scheduling algorithms within the distributed testing framework. These custom schedulers can implement different strategies for assigning tasks to workers, such as fair-share scheduling, priority-based scheduling, or specialized algorithms for specific workloads.

## Key Features

1. **Pluggable Scheduler Architecture**: Easily swap different scheduling algorithms without modifying core code
2. **Multiple Scheduling Strategies**: Support for various scheduling strategies within a single scheduler plugin
3. **Runtime Strategy Selection**: Change scheduling strategies dynamically during execution
4. **Standard Interface**: Consistent interface for all scheduler plugins
5. **Configuration Options**: Comprehensive configuration system for customizing scheduler behavior
6. **Metrics Collection**: Built-in metrics collection for scheduling performance analysis
7. **Seamless Integration**: Integration with existing coordinator and task scheduler

## Architecture

The custom scheduler extensibility system consists of the following components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Distributed Testing Framework                     │
│                                                                     │
│  ┌───────────────────┐         ┌───────────────────────────────┐    │
│  │    Coordinator    │◄────────┤    SchedulerPluginWrapper     │    │
│  └───────────────────┘         └───────────────┬───────────────┘    │
│                                                │                     │
│  ┌───────────────────┐         ┌───────────────▼───────────────┐    │
│  │  Task Scheduler   │◄────────┤   SchedulerPluginRegistry     │    │
│  └───────────────────┘         └───────────────┬───────────────┘    │
│                                                │                     │
└────────────────────────────────────────────────┼─────────────────────┘
                                                 │
                            ┌───────────────────┬┴──────────────────┐
                            │                   │                   │
                            ▼                   ▼                   ▼
              ┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────┐
              │FairnessScheduler    │ │PriorityScheduler│ │CustomScheduler  │
              │                     │ │                 │ │                 │
              │- fair_share         │ │- priority_based │ │- custom         │
              │- priority_based     │ │- deadline_driven│ │- specialized    │
              │- round_robin        │ │- load_balanced  │ │- domain_specific│
              └─────────────────────┘ └─────────────────┘ └─────────────────┘
```

## Creating a Custom Scheduler

To create a custom scheduler plugin, follow these steps:

1. Create a new Python file in the `distributed_testing/plugins/scheduler` directory
2. Implement the `SchedulerPluginInterface` or extend the `BaseSchedulerPlugin` class
3. Register your scheduler with the registry by implementing the required methods

### Example Scheduler Implementation

```python
from distributed_testing.plugins.scheduler.base_scheduler_plugin import BaseSchedulerPlugin
from distributed_testing.plugins.scheduler.scheduler_plugin_interface import SchedulingStrategy

class MyCustomScheduler(BaseSchedulerPlugin):
    """My custom scheduler implementation."""
    
    def __init__(self):
        super().__init__(
            name="MyCustomScheduler",
            version="1.0.0",
            description="My custom scheduler implementation",
            strategies=[
                SchedulingStrategy.ROUND_ROBIN,
                SchedulingStrategy.CUSTOM
            ]
        )
        
        # Add custom configuration options
        self.config.update({
            "my_custom_option": True,
            "another_option": 42
        })
    
    async def schedule_task(self, task_id, task_data, available_workers, worker_load):
        """Implement custom task scheduling logic."""
        # Your custom scheduling logic here
        
        # For example, select the worker with the lowest ID
        if available_workers:
            return min(available_workers.keys())
        
        return None
```

## Scheduling Strategies

The following scheduling strategies are supported by the scheduler interface:

| Strategy | Description |
|----------|-------------|
| `ROUND_ROBIN` | Simple round-robin assignment |
| `PRIORITY_BASED` | Assignment based on task priority |
| `HARDWARE_MATCH` | Matching tasks to hardware capabilities |
| `PERFORMANCE_BASED` | Assignment based on historical performance |
| `DEADLINE_DRIVEN` | Meeting task deadlines |
| `ENERGY_EFFICIENT` | Optimizing for energy efficiency |
| `LOAD_BALANCED` | Balancing load across workers |
| `FAIR_SHARE` | Fair resource allocation |
| `CUSTOM` | Custom scheduling algorithm |

## Using Custom Schedulers

To use a custom scheduler, you need to:

1. Create a `SchedulerCoordinator` instance
2. Initialize it with your coordinator
3. Activate your desired scheduler plugin
4. Set the desired scheduling strategy

```python
# Create scheduler coordinator
scheduler_coordinator = SchedulerCoordinator(coordinator)

# Initialize scheduler coordinator
await scheduler_coordinator.initialize()

# List available scheduler plugins
available_plugins = scheduler_coordinator.get_available_plugins()
print(f"Available scheduler plugins: {available_plugins}")

# Activate a scheduler plugin
await scheduler_coordinator.activate_scheduler("FairnessScheduler")

# Set the scheduling strategy
await scheduler_coordinator.set_strategy("fair_share")
```

## Available Scheduler Plugins

The following scheduler plugins are included by default:

### 1. FairnessScheduler

The `FairnessScheduler` implements fair resource allocation across users, projects, and priorities:

- Ensures no single user or project monopolizes resources
- Implements quotas and weights for users and projects
- Tracks historical resource usage for fairness
- Supports consecutive task limits for better interactivity

Strategies:
- `FAIR_SHARE`: Fair resource allocation
- `PRIORITY_BASED`: Priority-based scheduling
- `ROUND_ROBIN`: Simple round-robin scheduling

Configuration options:
- `fairness_window_hours`: Time window for historical usage calculation
- `enable_quotas`: Enable quota enforcement
- `recalculate_interval`: Interval to recalculate fair shares
- `max_consecutive_same_user`: Maximum consecutive tasks from the same user

## Running the Example

An example script is provided to demonstrate the custom scheduler extensibility:

```bash
# Run with default settings (FairnessScheduler with fair_share strategy)
python distributed_testing/examples/custom_scheduler_example.py

# Run with a specific scheduler and strategy
python distributed_testing/examples/custom_scheduler_example.py --scheduler FairnessScheduler --strategy priority_based

# Run with more workers and tasks
python distributed_testing/examples/custom_scheduler_example.py --num-workers 8 --num-tasks 50
```

## Configuration Options

Each scheduler plugin can define its own configuration options. Common options include:

- `max_tasks_per_worker`: Maximum number of concurrent tasks per worker
- `history_window_size`: Number of tasks to keep in performance history
- `detailed_logging`: Enable detailed scheduler logging

## Best Practices

When creating custom schedulers, follow these best practices:

1. **Implement Multiple Strategies**: Support multiple scheduling strategies for flexibility
2. **Collect Metrics**: Collect and expose metrics to help users understand scheduler performance
3. **Handle Edge Cases**: Gracefully handle edge cases like no available workers
4. **Use Appropriate Scoring**: Use a scoring system for complex scheduling decisions
5. **Document Behavior**: Document your scheduler's behavior and configuration options
6. **Test Thoroughly**: Test your scheduler with various task and worker configurations
7. **Provide Configuration**: Make scheduler behavior configurable through well-documented options

## Implementation Status

The Custom Scheduler Extensibility feature is now 100% complete as part of Phase 8 (Integration and Extensibility):

- ✅ Core scheduler plugin interface
- ✅ Scheduler plugin registry
- ✅ Base scheduler plugin implementation
- ✅ Fairness scheduler implementation
- ✅ Scheduler coordinator for integration
- ✅ Example implementation
- ✅ Documentation