# Distributed Testing Framework - Integration and Extensibility Guide

This guide covers the integration and extensibility capabilities of the Distributed Testing Framework, with a particular focus on the plugin architecture and integration with external systems.

## Table of Contents

1. [Plugin Architecture Overview](#plugin-architecture-overview)
2. [WebGPU/WebNN Resource Pool Integration](#webgpuwebnn-resource-pool-integration)
3. [CI/CD System Integration](#cicd-system-integration)
4. [Custom Scheduler Implementation](#custom-scheduler-implementation)
5. [Creating Your Own Plugins](#creating-your-own-plugins)
6. [Plugin Deployment](#plugin-deployment)
7. [API Reference](#api-reference)

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

### Key Features

- **Automatic Environment Detection**: Detects CI environment and configures automatically
- **Test Run Management**: Creates and manages test runs in CI systems
- **Status Updates**: Provides real-time status updates to CI systems
- **Result Reporting**: Generates comprehensive reports in multiple formats (JUnit XML, HTML, JSON)
- **PR Comments**: Automatically adds test result comments to pull requests
- **Artifact Management**: Uploads test artifacts to CI systems

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

## Custom Scheduler Implementation

The Custom Scheduler plugin extends the task scheduling capabilities of the Distributed Testing Framework with advanced scheduling algorithms and features.

### Key Features

- **Hardware-Aware Scheduling**: Matches tasks to workers based on hardware requirements
- **Priority-Based Scheduling**: Schedules tasks based on priority levels
- **Deadline-Driven Scheduling**: Increases priority as deadlines approach
- **Performance History**: Uses historical performance data for optimal task placement
- **Task Dependencies**: Supports complex task dependencies and DAG execution
- **Adaptive Scheduling**: Combines multiple scheduling strategies based on context
- **Performance Prediction**: Predicts task execution time for optimal scheduling

### Usage Example

```python
# Access custom scheduler plugin from coordinator
scheduler_plugin = coordinator.plugin_manager.get_plugins_by_type(PluginType.SCHEDULER)["CustomScheduler-1.0.0"]

# Check scheduler status
scheduler_status = scheduler_plugin.get_scheduler_status()
print(f"High Priority Tasks: {scheduler_status['high_priority_queue_size']}")
print(f"Normal Priority Tasks: {scheduler_status['normal_priority_queue_size']}")
print(f"Low Priority Tasks: {scheduler_status['low_priority_queue_size']}")
```

### Configuration Options

The Custom Scheduler plugin supports various configuration options:

- `max_tasks_per_worker`: Maximum tasks per worker (default: 5)
- `priority_levels`: Number of priority levels (default: 10)
- `enable_adaptive_scheduling`: Enable adaptive scheduling (default: true)
- `enable_deadline_scheduling`: Enable deadline-driven scheduling (default: true)
- `enable_hardware_matching`: Enable hardware-aware task assignment (default: true)
- `enable_performance_prediction`: Enable performance prediction (default: true)
- `prediction_confidence_threshold`: Confidence threshold for predictions (default: 0.7)
- `max_retry_attempts`: Maximum retry attempts for failed tasks (default: 3)
- `scheduler_interval`: Scheduler interval in seconds (default: 1.0)
- `detailed_logging`: Enable detailed scheduler logging (default: false)

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