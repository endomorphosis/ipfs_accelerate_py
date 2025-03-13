# Distributed Testing Framework - Integration and Extensibility Guide

This guide covers the integration and extensibility capabilities of the Distributed Testing Framework, with a particular focus on the plugin architecture and integration with external systems.

## Table of Contents

1. [Plugin Architecture Overview](#plugin-architecture-overview)
2. [WebGPU/WebNN Resource Pool Integration](#webgpuwebnn-resource-pool-integration)
3. [CI/CD System Integration](#cicd-system-integration)
4. [Custom Scheduler Implementation](#custom-scheduler-implementation)
5. [Creating Your Own Plugins](#creating-your-own-plugins)
6. [Plugin Deployment](#plugin-deployment)
7. [Running the Example](#running-the-example)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

## Plugin Architecture Overview

The Distributed Testing Framework features a flexible plugin architecture that allows extending its functionality without modifying the core codebase. The plugin system is based on the following key components:

### Plugin Types

The framework supports various plugin types, each providing specific functionality:

- **Scheduler**: Custom task scheduling algorithms
- **Task Executor**: Alternative task execution methods
- **Reporter**: Result reporting and visualization plugins
- **Notification**: Event notification and alerting
- **Monitoring**: System monitoring and metrics collection
- **Integration**: Integration with external systems
- **Security**: Custom security policies and authentication
- **Custom**: General-purpose plugins for extended functionality

### Hook System

Plugins can register hooks to respond to various events in the framework:

- **Coordinator Events**: Startup, shutdown
- **Task Events**: Creation, assignment, start, completion, failure, cancellation
- **Worker Events**: Registration, disconnection, failure, recovery
- **Recovery Events**: Start, completion, failure
- **Custom Events**: User-defined events

### Creating a Plugin

Creating a plugin involves extending the `Plugin` base class and implementing required methods:

```python
from distributed_testing.plugin_architecture import Plugin, PluginType, HookType

class MyPlugin(Plugin):
    def __init__(self):
        super().__init__(
            name="MyPlugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM
        )
        
        # Register hooks
        self.register_hook(HookType.COORDINATOR_STARTUP, self.on_coordinator_startup)
        
    async def initialize(self, coordinator) -> bool:
        # Initialize plugin with reference to coordinator
        self.coordinator = coordinator
        return True
        
    async def shutdown(self) -> bool:
        # Clean up resources
        return True
        
    async def on_coordinator_startup(self, coordinator):
        # Handle coordinator startup event
        print("Coordinator started!")
```

### Plugin Lifecycle

1. **Discovery**: The framework discovers available plugins in specified directories
2. **Loading**: Plugins are loaded and initialized with a reference to the coordinator
3. **Hook Registration**: Plugins register for events they wish to handle
4. **Event Handling**: Plugins respond to events via registered hooks
5. **Shutdown**: Plugins are given a chance to clean up resources during shutdown

## WebGPU/WebNN Resource Pool Integration

The WebGPU/WebNN Resource Pool Integration plugin connects the Distributed Testing Framework with the WebGPU/WebNN Resource Pool, enabling browser-based acceleration with fault tolerance for distributed tests.

### Key Features

- **Browser-Specific Optimizations**: Automatically routes tasks to optimal browsers (Firefox for audio, Chrome for vision, Edge for WebNN)
- **Connection Pooling**: Manages browser connections efficiently with lifecycle management
- **Fault Tolerance**: Provides automatic recovery from browser failures
- **Cross-Browser Model Sharding**: Distributes large models across multiple browsers
- **Performance History**: Tracks and analyzes performance trends for optimization
- **Metrics Collection**: Integrates with DuckDB for comprehensive performance tracking

### Usage Example

```python
# Get plugin instance from coordinator
webgpu_plugin = coordinator.plugin_manager.get_plugins_by_type(PluginType.INTEGRATION)["WebGPUResourcePool-1.0.0"]

# Get model from resource pool with fault tolerance
model = await webgpu_plugin.get_model(
    model_type="text_embedding",
    model_name="bert-base-uncased",
    hardware_preferences={'priority_list': ['webgpu', 'webnn', 'cpu']},
    fault_tolerance={
        'recovery_timeout': 30,
        'state_persistence': True,
        'failover_strategy': 'immediate'
    }
)

# Create sharded model execution
sharded_execution, exec_id = await webgpu_plugin.create_sharded_execution(
    model_name="llama-13b",
    num_shards=3,
    sharding_strategy="layer_balanced",
    fault_tolerance_level="high",
    recovery_strategy="coordinated"
)

# Run inference on sharded model
result = await sharded_execution.run_inference(inputs)

# Release resources when done
await webgpu_plugin.release_resources(exec_id=exec_id)
```

### Configuration Options

The WebGPU/WebNN Resource Pool Integration plugin supports various configuration options:

- `max_browser_connections`: Maximum number of browser connections (default: 4)
- `browser_preferences`: Browser preferences by model type
- `enable_fault_tolerance`: Enable fault tolerance features (default: true)
- `recovery_strategy`: Recovery strategy (progressive, immediate, coordinated)
- `state_sync_interval`: State synchronization interval in seconds (default: 5)
- `redundancy_factor`: Redundancy factor for critical operations (default: 2)
- `metric_collection`: Enable metrics collection (default: true)
- `recovery_timeout`: Maximum recovery time in seconds (default: 30)

## CI/CD System Integration

The CI/CD Integration plugin provides seamless integration with popular CI/CD systems:

- GitHub Actions
- Jenkins
- GitLab CI
- Azure DevOps

The integration is now more powerful with a standardized API interface that ensures consistent behavior across different CI/CD providers and makes it easier to add support for new systems.

### Key Features

- **Standardized API Interface**: Common interface for all CI/CD providers
- **Automatic Environment Detection**: Detects CI environment and configures automatically
- **Test Run Management**: Creates and manages test runs in CI systems
- **Status Updates**: Provides real-time status updates to CI systems
- **Result Reporting**: Generates comprehensive reports in multiple formats (JUnit XML, HTML, JSON)
- **PR Comments**: Automatically adds test result comments to pull requests
- **Artifact Management**: Uploads test artifacts to CI systems
- **Factory Pattern**: Easy creation of appropriate CI provider based on environment

### Usage Example

```python
# Access CI/CD plugin from coordinator
ci_plugin = coordinator.plugin_manager.get_plugins_by_type(PluginType.INTEGRATION)["CICDIntegration-1.0.0"]

# Check CI status
ci_status = ci_plugin.get_ci_status()
print(f"CI System: {ci_status['ci_system']}")
print(f"Test Run: {ci_status['test_run_id']}")
print(f"Status: {ci_status['test_run_status']}")
```

### Direct API Usage

```python
from distributed_testing.ci import CIProviderFactory

# Create appropriate CI provider based on configuration
provider = await CIProviderFactory.create_provider(
    "github",
    {
        "token": "YOUR_TOKEN",
        "repository": "owner/repo",
        "commit_sha": "1234567890abcdef"
    }
)

# Create a test run
test_run = await provider.create_test_run({
    "name": "Test Run Example",
    "build_id": "12345"
})

# Use provider through standardized interface
await provider.update_test_run(
    test_run["id"],
    {
        "status": "completed",
        "summary": {
            "total_tests": 10,
            "passed_tests": 8,
            "failed_tests": 2,
            "skipped_tests": 0,
            "duration_seconds": 25.5
        }
    }
)
```

### Configuration Options

The CI/CD Integration plugin supports various configuration options:

- `ci_system`: CI system to use (auto, github, jenkins, gitlab, azure)
- `api_token`: API token for authentication
- `update_interval`: Status update interval in seconds (default: 60)
- `update_on_completion_only`: Only update status on completion (default: false)
- `artifact_dir`: Directory for storing artifacts (default: "distributed_test_results")
- `result_format`: Result format (junit, json, html, all)
- `enable_status_updates`: Enable CI status updates (default: true)
- `enable_pr_comments`: Enable PR comments (default: true)
- `enable_artifacts`: Enable artifact upload (default: true)
- `detailed_logging`: Enable detailed logging (default: false)

For comprehensive documentation on CI/CD integration, including command-line usage, configuration examples for different CI/CD systems, and advanced usage scenarios, see the [CI/CD Integration Guide](docs/CI_CD_INTEGRATION_GUIDE.md).

## Custom Scheduler Extensibility

The Distributed Testing Framework now provides a comprehensive scheduler extensibility system that allows users to create, configure, and use custom scheduling algorithms. The system includes a standardized interface, a plugin registry, a base implementation, and example schedulers.

### Key Components

- **SchedulerPluginInterface**: Standardized interface that all scheduler plugins must implement
- **BaseSchedulerPlugin**: Base implementation with common functionality for scheduler plugins
- **SchedulerPluginRegistry**: Registry for discovering, loading, and managing scheduler plugins
- **SchedulerCoordinator**: Utility for integrating scheduler plugins with the coordinator
- **Scheduling Strategies**: Multiple strategies like round-robin, fair-share, load-balanced, etc.

### Included Scheduler Plugins

The framework includes several scheduler plugins out of the box:

1. **FairnessScheduler**: Fair resource allocation across users and projects
   - Ensures no single user or project monopolizes resources
   - Implements quotas and weights for users and projects
   - Tracks historical resource usage for fairness
   - Supports consecutive task limits for better interactivity

2. **CustomScheduler**: Advanced task scheduling with multiple features
   - Hardware-aware scheduling
   - Priority-based scheduling with dynamic priority adjustment
   - Deadline-driven scheduling
   - Performance history tracking
   - Worker specialization for specific task types

### Usage Example

```python
# Import scheduler components
from distributed_testing.plugins.scheduler.scheduler_coordinator import SchedulerCoordinator
from distributed_testing.plugins.scheduler.scheduler_plugin_interface import SchedulingStrategy

# Create scheduler coordinator
scheduler_coordinator = SchedulerCoordinator(coordinator)

# Initialize scheduler coordinator
await scheduler_coordinator.initialize()

# List available scheduler plugins
available_plugins = scheduler_coordinator.get_available_plugins()
print(f"Available scheduler plugins: {', '.join(available_plugins)}")

# Activate a scheduler plugin
await scheduler_coordinator.activate_scheduler("FairnessScheduler", {
    "fairness_window_hours": 24,
    "enable_quotas": True,
    "max_consecutive_same_user": 5
})

# Set the active scheduling strategy
await scheduler_coordinator.set_strategy("fair_share")

# Get metrics from the active scheduler
metrics = scheduler_coordinator.get_metrics()
print(f"Active users: {metrics['fairness']['active_users']}")
print(f"Active projects: {metrics['fairness']['active_projects']}")
print(f"Fairness score: {metrics['fairness'].get('fairness_score', 0.0):.2f}")
```

### Creating a Custom Scheduler

Creating a custom scheduler is easy using the `BaseSchedulerPlugin` class:

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
                SchedulingStrategy.LOAD_BALANCED
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

### Available Scheduling Strategies

The scheduler interface provides multiple scheduling strategies:

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

### Configuration Options

Each scheduler plugin defines its own configuration schema. Common options include:

- `max_tasks_per_worker`: Maximum tasks per worker (default: 5)
- `history_window_size`: Number of tasks to keep in performance history (default: 100)
- `detailed_logging`: Enable detailed scheduler logging (default: false)

The FairnessScheduler adds specialized options:

- `fairness_window_hours`: Time window for historical usage calculation (default: 24)
- `enable_quotas`: Enable quota enforcement (default: true)
- `recalculate_interval`: Interval to recalculate fair shares in seconds (default: 60)
- `max_consecutive_same_user`: Maximum consecutive tasks from the same user (default: 3)
- `enable_priority_boost`: Enable priority-based boosts (default: true)

For complete documentation on custom scheduler implementation, configuration options, and usage examples, see [plugins/scheduler/README.md](plugins/scheduler/README.md).

## Creating Your Own Plugins

Creating your own plugins involves these steps:

1. **Create a Plugin Class**: Extend the `Plugin` base class
2. **Register Hooks**: Register for events you want to handle
3. **Implement Core Methods**: Initialize, shutdown, and hook handlers
4. **Package Your Plugin**: Create a Python module with your plugin class
5. **Deploy Your Plugin**: Place your plugin in the coordinator's plugin directory

### Plugin Template

```python
#!/usr/bin/env python3
"""
Custom Plugin for Distributed Testing Framework
"""

import logging
from typing import Dict, List, Any

from distributed_testing.plugin_architecture import Plugin, PluginType, HookType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MyCustomPlugin(Plugin):
    """My custom plugin description."""
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="MyCustomPlugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM
        )
        
        # Default configuration
        self.config = {
            "option1": "value1",
            "option2": "value2"
        }
        
        # Register hooks
        self.register_hook(HookType.COORDINATOR_STARTUP, self.on_coordinator_startup)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        
        logger.info("MyCustomPlugin initialized")
    
    async def initialize(self, coordinator) -> bool:
        """Initialize the plugin with reference to the coordinator."""
        self.coordinator = coordinator
        logger.info("MyCustomPlugin initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        logger.info("MyCustomPlugin shutdown complete")
        return True
    
    async def on_coordinator_startup(self, coordinator):
        """Handle coordinator startup event."""
        logger.info("Coordinator startup detected")
    
    async def on_task_completed(self, task_id: str, result: Any):
        """Handle task completed event."""
        logger.info(f"Task {task_id} completed")
```

## Plugin Deployment

To deploy your plugins:

1. **Create a Plugin Directory**: Create a directory named `plugins` in the same directory as the coordinator script, or specify a custom directory.
2. **Add Your Plugins**: Place your plugin Python files in the plugin directory.
3. **Configure Plugin Loading**: Specify plugin directories when starting the coordinator.

```python
# Configure coordinator with custom plugin directory
coordinator = DistributedTestingCoordinator(
    db_path="benchmark_db.duckdb",
    enable_plugins=True,
    plugin_dirs=["plugins", "custom_plugins"]
)

# Start coordinator
await coordinator.start()
```

## Running the Example

The repository includes an example script that demonstrates the use of the plugins. This example creates and configures a notification plugin, simulates worker registration and task execution, and shows how to handle events through the plugin system.

### Prerequisites

- Python 3.8 or higher
- The Distributed Testing Framework must be installed

### Running the Example

```bash
# Navigate to the distributed_testing directory
cd /path/to/distributed_testing

# Run the example script
python examples/plugin_example.py
```

### Example Output

When you run the example, you should see output similar to the following:

```
2025-03-11 10:15:23,456 - __main__ - INFO - Starting coordinator...
2025-03-11 10:15:23,789 - __main__ - INFO - Discovered plugins: ['notification_plugin']
2025-03-11 10:15:24,123 - __main__ - INFO - Loaded notification plugin: SimpleNotification-1.0.0
2025-03-11 10:15:24,456 - __main__ - INFO - NOTIFICATION: Worker worker-001 registered with capabilities: {'hardware_type': 'gpu', 'cpu_cores': 8, 'memory_gb': 16, 'gpu_memory_gb': 8, 'supports_cuda': True, 'supports_webgpu': True, 'supports_webnn': True}
2025-03-11 10:15:24,789 - __main__ - INFO - NOTIFICATION: Task task-1 created with type: model_test
2025-03-11 10:15:25,123 - __main__ - INFO - NOTIFICATION: Task task-1 failed: Model test failed due to out of memory error
...
2025-03-11 10:15:30,456 - __main__ - INFO - Notification Summary:
2025-03-11 10:15:30,457 - __main__ - INFO - Total notifications: 11
2025-03-11 10:15:30,458 - __main__ - INFO -   info: 9
2025-03-11 10:15:30,459 - __main__ - INFO -   error: 1
2025-03-11 10:15:30,460 - __main__ - INFO -   warning: 1
2025-03-11 10:15:30,461 - __main__ - INFO - Last 3 notifications:
2025-03-11 10:15:30,462 - __main__ - INFO -   [info] Task task-5 created with type: model_test
2025-03-11 10:15:30,463 - __main__ - INFO -   [info] Task task-5 completed successfully
2025-03-11 10:15:30,464 - __main__ - INFO -   [warning] Worker worker-001 disconnected
2025-03-11 10:15:30,465 - __main__ - INFO - Shutting down coordinator...
2025-03-11 10:15:31,789 - __main__ - INFO - Demo completed successfully
```

### Understanding the Example

The example demonstrates several key aspects of the plugin architecture:

1. **Coordinator Initialization**: Creating a coordinator with plugin support
2. **Plugin Discovery and Loading**: Discovering and loading plugins from the plugin directory
3. **Plugin Configuration**: Setting configuration options for plugins
4. **Hook Invocation**: Simulating events that trigger plugin hooks
5. **Event Handling**: Handling events in the plugin and taking appropriate action
6. **Plugin Shutdown**: Properly shutting down plugins when the coordinator shuts down

## API Reference

### Plugin Base Class

```python
class Plugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, name: str, version: str, plugin_type: PluginType):
        """Initialize plugin."""
        
    @abstractmethod
    async def initialize(self, coordinator) -> bool:
        """Initialize the plugin with reference to the coordinator."""
        
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        
    def register_hook(self, hook_type: HookType, callback: Callable) -> bool:
        """Register a hook callback."""
        
    def unregister_hook(self, hook_type: HookType, callback: Callable) -> bool:
        """Unregister a hook callback."""
        
    async def invoke_hook(self, hook_type: HookType, *args, **kwargs) -> List[Any]:
        """Invoke all callbacks registered for a hook."""
        
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the plugin."""
        
    def enable(self) -> bool:
        """Enable the plugin."""
        
    def disable(self) -> bool:
        """Disable the plugin."""
        
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
```

### Plugin Manager

```python
class PluginManager:
    """Manager for loading, configuring, and invoking plugins."""
    
    def __init__(self, coordinator, plugin_dirs: List[str] = None):
        """Initialize the plugin manager."""
        
    async def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directories."""
        
    async def load_plugin(self, module_name: str) -> Optional[str]:
        """Load a plugin by module name."""
        
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        
    async def invoke_hook(self, hook_type: HookType, *args, **kwargs) -> List[Any]:
        """Invoke all callbacks registered for a hook."""
        
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID."""
        
    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, Plugin]:
        """Get all plugins of a specific type."""
        
    def get_all_plugins(self) -> Dict[str, Plugin]:
        """Get all loaded plugins."""
```

### Hook Types

```python
class HookType(Enum):
    """Hook points where plugins can be invoked."""
    
    # Coordinator hooks
    COORDINATOR_STARTUP = "coordinator_startup"
    COORDINATOR_SHUTDOWN = "coordinator_shutdown"
    
    # Task hooks
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    
    # Worker hooks
    WORKER_REGISTERED = "worker_registered"
    WORKER_DISCONNECTED = "worker_disconnected"
    WORKER_FAILED = "worker_failed"
    WORKER_RECOVERED = "worker_recovered"
    
    # Recovery hooks
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_FAILED = "recovery_failed"
```

### Plugin Types

```python
class PluginType(Enum):
    """Types of plugins supported by the framework."""
    
    SCHEDULER = "scheduler"
    TASK_EXECUTOR = "task_executor"
    REPORTER = "reporter"
    NOTIFICATION = "notification"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    SECURITY = "security"
    CUSTOM = "custom"
```

## Troubleshooting

### Common Issues

#### Plugin Not Found

**Symptom**: `Plugin module not found` error when trying to load a plugin.

**Solution**:
- Ensure the plugin file is in the correct directory
- Check that the plugin file name matches the module name
- Verify that the plugin directory is correctly specified in plugin_dirs

#### Hook Not Called

**Symptom**: A hook handler in your plugin is not being called for an event.

**Solution**:
- Verify that you registered the hook with the correct HookType
- Check that the plugin is enabled
- Ensure the hook handler signature matches the expected parameters
- Add debug logging in your hook handler to confirm it's registered

#### Plugin Initialization Failed

**Symptom**: `Failed to initialize plugin` error when loading a plugin.

**Solution**:
- Check for errors in the plugin's initialize method
- Ensure the plugin has the required dependencies
- Verify that the plugin can access the resources it needs

### Debugging Tips

1. **Enable Detailed Logging**: Set logging level to DEBUG for more detailed information
2. **Check Plugin Registration**: Verify that hooks are properly registered in the plugin's `__init__` method
3. **Inspect Plugin Configuration**: Print plugin configuration to ensure it's correctly set
4. **Test Hooks Individually**: Trigger hooks manually to test specific functionality
5. **Plugin Isolation**: Test plugins in isolation to identify interaction issues

### Getting Help

If you encounter issues that aren't covered here, please:
1. Check the framework documentation for additional information
2. Review the plugin API reference to ensure correct usage
3. Examine the example plugins for best practices
4. Create detailed bug reports with steps to reproduce the issue