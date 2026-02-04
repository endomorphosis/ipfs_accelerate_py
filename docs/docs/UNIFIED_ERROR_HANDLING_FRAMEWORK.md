# Unified Error Handling Framework

**Date:** March 10, 2025  
**Version:** 1.0  
**Status:** Implemented

## Overview

The Unified Error Handling Framework provides a comprehensive, consistent approach to error handling across all components of the IPFS Accelerate Python Framework. It standardizes error categorization, reporting, recovery strategies, and logging to improve robustness and maintainability.

## Core Components

### Error Categories

The framework defines standardized error categories:

```python
class ErrorCategories:
    NETWORK = "network_error"         # Network and connection issues
    TIMEOUT = "timeout_error"         # Timeouts in various operations
    INPUT = "input_error"             # Invalid input parameters
    RESOURCE = "resource_error"       # Memory and resource constraints
    IO = "io_error"                   # File and I/O operations
    DATA = "data_error"               # Data format and parsing
    DEPENDENCY = "dependency_error"   # Missing or incompatible dependencies
    INTERNAL = "internal_error"       # Internal system errors
    VALIDATION = "validation_error"   # Validation failures
    PERMISSION = "permission_error"   # Permission and access issues
    NOT_FOUND = "not_found_error"     # Resources not found
    INITIALIZATION = "initialization_error"  # Initialization failures
    HARDWARE = "hardware_error"       # Hardware-related issues
    UNKNOWN = "unknown_error"         # Unclassified errors
```

### ErrorHandler Class

The `ErrorHandler` class provides methods for:

1. **Error categorization**: Map exceptions to standardized categories
2. **Error formatting**: Create standardized error objects with rich context
3. **Error handling**: Log errors with appropriate severity and context
4. **Recovery strategies**: Determine if errors are recoverable and how
5. **Error responses**: Generate standardized error responses for APIs

### Error Handling Decorators

The framework provides decorators for adding standardized error handling:

1. **`@handle_errors`**: For synchronous functions
2. **`@handle_async_errors`**: For asynchronous functions
3. **`@with_retry`**: Add automatic retry with exponential backoff
4. **`@validate_dependencies`**: Check required dependencies before execution

## Key Features

### 1. Hierarchical Error Categorization

All errors are categorized into a standardized set of categories, making it easier to:
- Apply consistent handling strategies
- Generate appropriate error messages
- Determine if errors are recoverable
- Log at appropriate severity levels

### 2. Rich Context Collection

Error handling automatically collects:
- Source file, line number, and function name where error occurred
- Function arguments and parameters (safely truncated)
- Class and module information
- Custom context provided at error site

### 3. Standardized Error Response Format

All error responses follow a consistent format:
```json
{
  "status": "error",
  "error": "Error message",
  "error_type": "TypeError",
  "error_category": "input_error",
  "recoverable": true,
  "timestamp": 1714329245000,
  "error_location": {
    "file": "/path/to/file.py",
    "line": 123,
    "function": "process_data"
  }
}
```

### 4. Automatic Recovery Strategies

The framework includes built-in recovery strategies for different error types:

| Error Category | Retry Delay | Max Retries | Strategy Description |
|----------------|-------------|-------------|----------------------|
| `NETWORK`      | 2.0s        | 5           | Wait and retry with exponential backoff |
| `TIMEOUT`      | 3.0s        | 3           | Retry with increased timeout |
| `RESOURCE`     | 5.0s        | 2           | Free resources and retry |
| `DATA`         | 0.5s        | 2           | Validate data and retry |

### 5. Dependency Validation

Automatic validation of required dependencies:
```python
@validate_dependencies("websockets", "selenium")
def initialize_browser():
    # This function will only execute if all dependencies are available
    # Otherwise, it returns a standardized error response
```

## Usage Examples

### Basic Usage with Try/Except

```python
from fixed_web_platform.unified_framework.error_handling import ErrorHandler

# Create error handler instance
error_handler = ErrorHandler()

try:
    # Code that might raise an exception
    result = perform_operation()
except Exception as e:
    # Handle with context
    error_response = error_handler.handle_error(
        e, 
        context={"operation": "data_processing", "input_size": len(data)}
    )
    
    # Return standardized error response
    return error_handler.create_error_response(e, context)
```

### Using Error Handling Decorators

```python
from fixed_web_platform.unified_framework.error_handling import handle_errors, handle_async_errors

# For synchronous functions
@handle_errors
def process_data(data):
    # If an exception occurs, it will be caught, logged,
    # and a standardized error response will be returned
    result = complex_processing(data)
    return result

# For asynchronous functions
@handle_async_errors
async def fetch_data(url):
    # Async function with automatic error handling
    response = await http_client.get(url)
    data = await response.json()
    return data
```

### Using Retry Decorator

```python
from fixed_web_platform.unified_framework.error_handling import with_retry

# Add automatic retry with exponential backoff
@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
async def connect_to_service(service_url):
    # This function will be retried up to 3 times with exponential backoff
    # if it raises a recoverable exception
    connection = await establish_connection(service_url)
    return connection
```

### Dependency Validation

```python
from fixed_web_platform.unified_framework.error_handling import validate_dependencies

# Check for required dependencies
@validate_dependencies("websockets", "selenium", "duckdb")
def initialize_web_test_environment():
    # This function will only run if all dependencies are available
    import websockets
    import selenium
    import duckdb
    
    # Initialize environment
    # ...
```

### Combining Multiple Decorators

```python
from fixed_web_platform.unified_framework.error_handling import (
    handle_async_errors, with_retry, validate_dependencies
)

# Combine multiple decorators (applied from bottom to top)
@handle_async_errors
@with_retry(max_retries=3)
@validate_dependencies("websockets")
async def websocket_connect(url):
    # This function will:
    # 1. Check that websockets is installed
    # 2. Retry up to 3 times if connection fails with recoverable error
    # 3. Handle any other errors with standardized error response
    import websockets
    connection = await websockets.connect(url)
    return connection
```

## Integration with WebSocket Bridge

The WebSocket Bridge (in `fixed_web_platform/websocket_bridge.py`) has been enhanced to use the new error handling framework:

```python
from fixed_web_platform.unified_framework.error_handling import (
    ErrorHandler, handle_errors, handle_async_errors, with_retry,
    validate_dependencies, ErrorCategories
)

class WebSocketBridge:
    # ...
    
    # Use with_retry decorator for built-in retry capability
    @with_retry(max_retries=2, initial_delay=0.1, backoff_factor=2.0)
    async def send_message(self, message, timeout=None, retry_attempts=2):
        # Implementation with standardized error handling
        # ...
```

## Implementation Status

The Unified Error Handling Framework has been implemented in:
- `fixed_web_platform/unified_framework/error_handling.py`: Core implementation
- `fixed_web_platform/websocket_bridge.py`: Enhanced with framework integration

### Next Steps for Integration

The following components should be updated to use the new framework:
1. WebGPU/WebNN implementation modules
2. Database integration components
3. Hardware detection modules
4. API client implementations
5. Model loading and inference components

## Conclusion

The Unified Error Handling Framework provides a consistent, robust approach to error handling across all IPFS Accelerate components. By standardizing error categorization, reporting, recovery strategies, and logging, it improves code quality, maintainability, and user experience.

## Related Documentation

- [ERROR_CODE_REFERENCE.md](ERROR_CODE_REFERENCE.md): Comprehensive reference for all error codes
- [ERROR_HANDLING_GUIDE.md](ERROR_HANDLING_GUIDE.md): Detailed guide for JavaScript web platform error handling
- [RESOURCE_POOL_ERROR_RECOVERY.md](../fixed_web_platform/RESOURCE_POOL_ERROR_RECOVERY.md): Error recovery for resource pool
- [API_DOCUMENTATION.md](../API_DOCUMENTATION.md): API error handling documentation