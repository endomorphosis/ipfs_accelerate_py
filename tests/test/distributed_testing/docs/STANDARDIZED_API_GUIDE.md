# Standardized API Guide for Distributed Testing

This guide explains the standardized API interfaces provided by the Distributed Testing Framework, enabling consistent behavior across different integrations and making it easier to extend functionality.

## Table of Contents

1. [Introduction](#introduction)
2. [API Design Principles](#api-design-principles)
3. [CI/CD Provider API](#cicd-provider-api)
4. [Plugin API](#plugin-api)
5. [WebGPU/WebNN Resource Pool API](#webgpuwebnn-resource-pool-api)
6. [Implementing Custom Providers](#implementing-custom-providers)
7. [API Evolution and Versioning](#api-evolution-and-versioning)
8. [Best Practices](#best-practices)

## Introduction

The Distributed Testing Framework provides standardized APIs to ensure consistent behavior across different components, providers, and integrations. These APIs follow a common design pattern and make it easy to:

- Switch between different implementations of a service
- Add support for new systems without changing existing code
- Create a consistent experience regardless of underlying technology
- Test components in isolation with mocks that follow the same interface
- Extend the framework with custom functionality

The most important standardized APIs are:

- **CI/CD Provider API**: Interface for integrating with CI/CD systems
- **Plugin API**: Interface for extending the framework with plugins
- **WebGPU/WebNN Resource Pool API**: Interface for browser-based acceleration

## API Design Principles

All standardized APIs in the framework follow these core principles:

1. **Interface-based Design**: Each API defines an abstract interface that concrete implementations must follow
2. **Factory Pattern**: Factories are provided to create appropriate implementations based on configuration
3. **Async/Await Support**: All APIs use async/await for asynchronous operations
4. **Consistent Error Handling**: APIs provide consistent error reporting and handling mechanisms
5. **Self-contained**: Each API is self-contained with minimal dependencies
6. **Extensible**: APIs are designed to be extended with new functionality
7. **Well-documented**: All APIs have comprehensive documentation
8. **Testable**: All APIs can be easily tested with mock implementations

## CI/CD Provider API

The CI/CD Provider API provides a standardized interface for integrating with various CI/CD systems.

### Core Components

- **CIProviderInterface**: Abstract base class for all CI providers
- **TestRunResult**: Standardized representation of test results
- **CIProviderFactory**: Factory for creating provider instances

### Using the CI/CD Provider API

```python
from distributed_testing.ci import CIProviderFactory, TestRunResult

async def run_tests_with_ci_reporting():
    # Create appropriate provider based on environment
    provider = await CIProviderFactory.create_provider(
        "github",
        {
            "token": os.environ.get("GITHUB_TOKEN"),
            "repository": os.environ.get("GITHUB_REPOSITORY")
        }
    )
    
    # Create a test run
    test_run = await provider.create_test_run({
        "name": "Example Test Run",
        "build_id": os.environ.get("GITHUB_RUN_ID")
    })
    
    try:
        # Run your tests here...
        test_results = run_my_tests()
        
        # Report results using standardized format
        result = TestRunResult(
            test_run_id=test_run["id"],
            status="completed" if test_results.success else "failed",
            total_tests=test_results.total,
            passed_tests=test_results.passed,
            failed_tests=test_results.failed,
            skipped_tests=test_results.skipped,
            duration_seconds=test_results.duration
        )
        
        # Update test run with results
        await provider.update_test_run(
            test_run["id"],
            {
                "status": result.status,
                "summary": result.to_dict()
            }
        )
        
        # Add comment to PR if applicable
        if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
            pr_number = extract_pr_number()
            await provider.add_pr_comment(
                pr_number,
                f"## Test Results\n\n{result.passed_tests}/{result.total_tests} tests passed"
            )
            
    finally:
        # Clean up resources
        await provider.close()
```

## Plugin API

The Plugin API provides a standardized way to extend the framework with custom functionality.

### Core Components

- **Plugin**: Abstract base class for all plugins
- **PluginType**: Enum defining supported plugin types
- **HookType**: Enum defining available hook points
- **PluginManager**: Manager for loading and invoking plugins

### Using the Plugin API

```python
from distributed_testing.plugin_architecture import Plugin, PluginType, HookType

class MyReporterPlugin(Plugin):
    """Custom reporter plugin."""
    
    def __init__(self):
        super().__init__(
            name="MyReporter",
            version="1.0.0",
            plugin_type=PluginType.REPORTER
        )
        
        # Register hooks
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        
    async def initialize(self, coordinator) -> bool:
        """Initialize the plugin."""
        self.coordinator = coordinator
        self.results = []
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        await self.save_results()
        return True
        
    async def on_task_completed(self, task_id: str, result: Any):
        """Handle task completed event."""
        self.results.append({
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    async def save_results(self):
        """Save results to a file."""
        with open("test_results.json", "w") as f:
            json.dump(self.results, f)
```

## WebGPU/WebNN Resource Pool API

The WebGPU/WebNN Resource Pool API provides a standardized interface for browser-based hardware acceleration.

### Core Components

- **ResourcePoolInterface**: Abstract base class for resource pools
- **ModelExecutorInterface**: Abstract base class for model executors
- **ResourceType**: Enum defining resource types
- **ResourcePoolFactory**: Factory for creating resource pool instances

### Using the Resource Pool API

```python
from distributed_testing.resource_pool import ResourcePoolFactory, ResourceType

async def run_model_with_resource_pool():
    # Create resource pool
    pool = await ResourcePoolFactory.create_pool(
        pool_type="browser",
        config={
            "max_connections": 4,
            "browser_preferences": {
                "audio": "firefox",
                "vision": "chrome",
                "text": "edge"
            }
        }
    )
    
    try:
        # Get model executor for BERT
        executor = await pool.get_model_executor(
            model_name="bert-base-uncased",
            resource_type=ResourceType.WEBGPU,
            preferences={
                "max_memory": 4096,
                "precision": "fp16"
            }
        )
        
        # Run inference
        result = await executor.run_inference({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        
        # Process result
        return process_result(result)
        
    finally:
        # Release resources
        await pool.release_all()
```

## Implementing Custom Providers

To implement a custom provider for any of the standardized APIs, follow these steps:

1. **Identify the Interface**: Determine which interface you need to implement
2. **Create Implementation Class**: Create a class that extends the abstract base class
3. **Implement Required Methods**: Implement all required methods in the interface
4. **Register with Factory**: Register your implementation with the appropriate factory
5. **Test Your Implementation**: Create tests to verify your implementation works correctly

Example for a custom CI provider:

```python
from distributed_testing.ci.api_interface import CIProviderInterface

class CircleCIProvider(CIProviderInterface):
    """CircleCI provider implementation."""
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with configuration."""
        self.token = config.get("token")
        self.project = config.get("project")
        
        if not self.token or not self.project:
            return False
            
        self.session = aiohttp.ClientSession(headers={
            "Circle-Token": self.token,
            "Accept": "application/json"
        })
        
        return True
    
    # Implement other required methods...
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()

# Register with factory
from distributed_testing.ci import CIProviderFactory
CIProviderFactory.register_provider("circleci", CircleCIProvider)
```

## API Evolution and Versioning

The standardized APIs follow semantic versioning principles:

- **MAJOR**: Incompatible API changes (breaking changes)
- **MINOR**: Added functionality in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes

API evolution guidelines:

1. **Backward Compatibility**: New versions should be backward compatible when possible
2. **Interface Extension**: Extend interfaces with new methods rather than modifying existing ones
3. **Default Implementations**: Provide default implementations for new methods when possible
4. **Deprecation Process**: Use deprecation warnings before removing functionality

## Best Practices

When working with the standardized APIs, follow these best practices:

1. **Use Factories**: Always create implementations through factories
2. **Handle Errors**: Implement proper error handling for all API calls
3. **Resource Cleanup**: Always clean up resources using the close method or context managers
4. **Async/Await**: Use proper async/await patterns for asynchronous operations
5. **Testing**: Write tests for your custom implementations
6. **Configuration**: Use environment variables for configuration when possible
7. **Security**: Never hardcode credentials or sensitive information
8. **Logging**: Implement appropriate logging for debugging
9. **Documentation**: Document your custom implementations

For additional guidance and examples, refer to the specific documentation for each API:

- [CI/CD Integration Guide](CI_CD_INTEGRATION_GUIDE.md): Detailed documentation for CI/CD integration
- [Plugin Architecture Guide](../README_PLUGIN_ARCHITECTURE.md): Comprehensive guide to the plugin architecture
- [Resource Pool Guide](../WEB_RESOURCE_POOL_INTEGRATION.md): Documentation for the WebGPU/WebNN resource pool
- [External Systems API Reference](EXTERNAL_SYSTEMS_API_REFERENCE.md): Comprehensive reference for the External Systems API

The External Systems API Reference provides detailed documentation for all external system connectors, including JIRA, Slack, TestRail, Prometheus, Email, and MS Teams. It includes comprehensive examples, best practices, and troubleshooting guidance.