# Enhanced Error Handling Implementation Guide

This guide provides instructions for implementing and using the enhanced error handling system in the Distributed Testing Framework.

## Overview

The enhanced error handling system adds comprehensive error management capabilities to the distributed testing framework, including:

1. **Error categorization**: Automatically identifies and categorizes errors
2. **Intelligent recovery strategies**: Applies appropriate recovery actions based on error type
3. **Automatic retries**: Configurable retry policies with exponential backoff
4. **Error pattern detection**: Identifies recurring error patterns across multiple tasks
5. **Comprehensive reporting**: Detailed error tracking and reporting

## Implementation Status

The implementation includes the following components:

- ✅ `distributed_error_handler.py`: Core error handling system
- ✅ `coordinator_error_integration.py`: Integration with the coordinator
- ✅ Tests for all components

## Installation Steps

1. **Copy the implementation files to your project**:

   Ensure the following files are present in your project:
   - `/duckdb_api/distributed_testing/distributed_error_handler.py`
   - `/duckdb_api/distributed_testing/coordinator_error_integration.py`
   - `/duckdb_api/distributed_testing/tests/test_distributed_error_handler.py`
   - `/duckdb_api/distributed_testing/tests/test_coordinator_error_integration.py`

2. **Integrate with the coordinator**:

   Add the following code to the coordinator initialization:

   ```python
   from duckdb_api.distributed_testing.coordinator_error_integration import integrate_error_handler
   
   # After coordinator is initialized
   coordinator = integrate_error_handler(coordinator)
   ```

3. **Run tests to verify implementation**:

   ```bash
   python -m unittest duckdb_api/distributed_testing/tests/test_distributed_error_handler.py
   python -m unittest duckdb_api/distributed_testing/tests/test_coordinator_error_integration.py
   ```

## Usage Guide

### Error Handling in the Coordinator

The enhanced error handling is automatically applied to the coordinator's error handling methods:

- `coordinator.handle_task_error`: Enhanced error handling for task failures
- `coordinator.handle_worker_error`: Enhanced error handling for worker failures

No changes to your code are needed to use these enhanced methods.

### Custom Error Handling

You can directly use the error handler for custom error handling:

```python
from duckdb_api.distributed_testing.distributed_error_handler import DistributedErrorHandler

# Create error handler
error_handler = DistributedErrorHandler()

# Handle an error
error = {
    "type": "ConnectionError",
    "message": "Connection refused",
    "traceback": "...",
    "hardware_context": {
        "hardware_type": "cuda",
        "device_id": 0
    }
}

context = {
    "worker_id": "worker1",
    "hardware_type": "cuda",
    "attempt_count": 1
}

result = error_handler.handle_error("task123", error, context)

# Check result
if result["retry"]:
    print(f"Retry recommended with delay: {result['retry_delay']} seconds")
    
if result["recovery_action"]:
    print(f"Recovery actions: {result['recovery_action']['actions_taken']}")
```

### Configuring Error Handling

You can customize error handling behavior by providing a configuration:

```python
config = {
    # Custom error category mapping
    "error_categories": {
        "MyCustomError": ErrorCategory.RESOURCE_EXHAUSTED
    },
    
    # Custom retry policies
    "retry_policies": {
        "RESOURCE_EXHAUSTED": {
            "max_retries": 10,
            "retry_delay_seconds": 30,
            "retry_backoff_factor": 1.5
        }
    },
    
    # Custom error message patterns
    "error_patterns": {
        r"(?i)custom\s+error\s+pattern": "TEST_ASSERTION_ERROR"
    },
    
    # Similarity threshold for error aggregation
    "similarity_threshold": 0.75
}

# Create error handler with custom configuration
error_handler = DistributedErrorHandler(config)
```

## Recovery Strategies

The system includes several recovery strategies:

1. **ResourceRecoveryStrategy**: For resource-related errors
   - Requests resource cleanup
   - Marks resources as unavailable
   - Reallocates tasks to different resources

2. **NetworkRecoveryStrategy**: For network-related errors
   - Increases timeouts for retries
   - Requests reconnection
   - Performs failover to alternate servers

3. **HardwareRecoveryStrategy**: For hardware-related errors
   - Marks hardware as unavailable
   - Reallocates tasks to alternative hardware
   - Updates hardware requirements

4. **WorkerRecoveryStrategy**: For worker-related errors
   - Marks workers as unavailable or slow
   - Reassigns tasks to different workers
   - Adjusts worker priorities

5. **TestExecutionRecoveryStrategy**: For test execution errors
   - Checks and resolves dependencies
   - Records test failures and errors

## Error Categories

The system categorizes errors into the following types:

### Resource Errors
- `RESOURCE_EXHAUSTED`: Out of memory or resources
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `RESOURCE_UNAVAILABLE`: Resource temporarily unavailable

### Network Errors
- `NETWORK_TIMEOUT`: Connection or request timed out
- `NETWORK_CONNECTION_ERROR`: Failed to establish connection
- `NETWORK_SERVER_ERROR`: Server returned an error response

### Test Execution Errors
- `TEST_ASSERTION_ERROR`: Test assertion failed
- `TEST_IMPORT_ERROR`: Failed to import module
- `TEST_DEPENDENCY_ERROR`: Missing or circular dependency
- `TEST_SYNTAX_ERROR`: Syntax error in test

### Hardware Errors
- `HARDWARE_NOT_AVAILABLE`: Required hardware not available
- `HARDWARE_MISMATCH`: Hardware does not match requirements
- `HARDWARE_COMPATIBILITY_ERROR`: Hardware compatibility issue

### Worker Errors
- `WORKER_DISCONNECTED`: Worker disconnected unexpectedly
- `WORKER_TIMEOUT`: Worker timed out while processing
- `WORKER_CRASHED`: Worker crashed during execution

### System Errors
- `SYSTEM_CRASH`: System crash or fatal error
- `SYSTEM_RESOURCE_LIMIT`: System resource limit reached

### Other
- `UNKNOWN`: Unknown or uncategorized error

## Extending the System

### Adding Custom Error Categories

To add a custom error category:

```python
from duckdb_api.distributed_testing.distributed_error_handler import ErrorCategory
import enum

# Create a subclass with additional categories
class CustomErrorCategory(ErrorCategory):
    CUSTOM_ERROR = "custom_error"
```

### Adding Custom Recovery Strategies

To add a custom recovery strategy:

```python
from duckdb_api.distributed_testing.distributed_error_handler import RecoveryStrategy, ErrorCategory

class CustomRecoveryStrategy(RecoveryStrategy):
    """Custom recovery strategy."""
    
    def recover(self, task_id, error, context):
        """Implement custom recovery logic."""
        recovery_result = {
            "success": False,
            "strategy": "custom",
            "actions_taken": [],
            "retry_recommended": False
        }
        
        # Implement custom recovery logic
        recovery_result["actions_taken"].append("custom_action")
        recovery_result["success"] = True
        recovery_result["retry_recommended"] = True
        
        return recovery_result
    
    def is_applicable(self, error_category, hardware_type=None):
        """Determine if this strategy applies to the error."""
        return error_category in [
            ErrorCategory.CUSTOM_ERROR
        ]
```

### Integrating Custom Strategies

To integrate custom strategies with the error handler:

```python
handler = DistributedErrorHandler()
handler.recovery_strategies.append(CustomRecoveryStrategy())
```

## Best Practices

1. **Comprehensive Error Information**: Include as much context as possible in error reports
2. **Appropriate Retries**: Configure retry policies based on error characteristics
3. **Graceful Degradation**: Use recovery strategies to maintain system functionality
4. **Error Analysis**: Regularly review error patterns for system improvement

## Troubleshooting

### Common Issues

1. **Incorrect Error Categorization**: 
   - Check error type and message patterns
   - Add custom patterns for specific error messages

2. **Recovery Action Failures**:
   - Verify coordinator and worker are properly synced
   - Check that recovery actions are compatible with current system state

3. **Excessive Retries**:
   - Adjust retry policies for specific error categories
   - Add exponential backoff and jitter to avoid retry storms

### Logging

Enable verbose logging for debugging:

```python
import logging
logging.getLogger("distributed_error_handler").setLevel(logging.DEBUG)
```

## Additional Resources

- `ENHANCED_ERROR_HANDLING_IMPLEMENTATION.md`: Detailed implementation specification
- `test_distributed_error_handler.py`: Examples of error handling in action
- `test_coordinator_error_integration.py`: Examples of coordinator integration