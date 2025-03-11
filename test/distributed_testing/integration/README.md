# Distributed Testing Framework - Integration Plugins

This directory contains integration plugins for the Distributed Testing Framework, enabling seamless connections with external systems and services.

## Available Plugins

### WebGPU/WebNN Resource Pool Integration

**File**: `webgpu_resource_pool_plugin.py`

This plugin provides integration between the Distributed Testing Framework and the WebGPU/WebNN Resource Pool, enabling browser-based acceleration with fault tolerance for distributed tests.

Key features:
- Browser-specific optimizations for different model types
- Connection pooling for efficient browser management
- Fault tolerance with automatic recovery
- Cross-browser model sharding for large models
- Performance history tracking and analysis

### CI/CD Integration

**File**: `ci_cd_integration_plugin.py`

This plugin integrates the Distributed Testing Framework with CI/CD systems like GitHub Actions, Jenkins, GitLab CI, and Azure DevOps.

Key features:
- Automatic CI environment detection
- Test run management in CI systems
- Status updates to CI systems
- Artifact management
- PR comments with test results
- Multi-format reporting (JUnit XML, HTML, JSON)

### Custom Scheduler

**File**: `custom_scheduler_plugin.py`

This plugin enhances the task scheduling capabilities with advanced algorithms and features.

Key features:
- Hardware-aware scheduling
- Priority-based scheduling
- Deadline-driven scheduling
- Performance history-based scheduling
- Task dependency resolution
- Adaptive scheduling strategies

## Usage

To use these plugins, include this directory in the plugin directories when initializing the coordinator:

```python
coordinator = DistributedTestingCoordinator(
    db_path="benchmark_db.duckdb",
    enable_plugins=True,
    plugin_dirs=["distributed_testing/integration"]
)
```

Refer to the [Integration and Extensibility Guide](../INTEGRATION_EXTENSIBILITY_GUIDE.md) for comprehensive documentation and examples.