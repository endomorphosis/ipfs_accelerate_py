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
**Client Modules**: Located in `../ci` directory

This plugin integrates the Distributed Testing Framework with CI/CD systems like GitHub Actions, Jenkins, GitLab CI, and Azure DevOps. It includes specialized client implementations for each supported CI/CD system.

Key features:
- Automatic CI environment detection and configuration
- Test run management in CI systems with real-time status updates
- Comprehensive reporting in multiple formats (JUnit XML, HTML, JSON)
- Pull request/merge request comments with detailed test results
- Artifact management and upload to CI systems
- Graceful degradation when CI systems are unavailable

Example usage:

```python
from distributed_testing.plugin_architecture import PluginType
from distributed_testing.integration.ci_cd_integration_plugin import CICDIntegrationPlugin

# Access CI/CD plugin from coordinator
ci_plugin = coordinator.plugin_manager.get_plugins_by_type(PluginType.INTEGRATION)["CICDIntegration-1.0.0"]

# Check CI status
ci_status = ci_plugin.get_ci_status()
print(f"CI System: {ci_status['ci_system']}")
print(f"Test Run: {ci_status['test_run_id']}")
print(f"Status: {ci_status['test_run_status']}")
```

Configuration options:

| Option | Default | Description |
|--------|---------|-------------|
| `ci_system` | auto | CI system to use (auto, github, jenkins, gitlab, azure) |
| `api_token` | None | API token for authentication |
| `update_interval` | 60 | Status update interval in seconds |
| `result_format` | junit | Result format (junit, json, html, all) |
| `enable_pr_comments` | true | Enable PR comments with results |

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