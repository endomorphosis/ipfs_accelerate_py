# Enhanced Worker Error Reporting Guide

This guide documents the Enhanced Worker Error Reporting system for the Distributed Testing Framework, which provides comprehensive error context, categorization, and system metrics to help coordinators make more intelligent recovery decisions.

![Error Reporting](https://img.shields.io/badge/Error%20Reporting-Enhanced-success)
![System Metrics](https://img.shields.io/badge/System%20Metrics-Comprehensive-success)
![Hardware Context](https://img.shields.io/badge/Hardware%20Context-Detailed-success)

## Features

- **Standardized Error Categorization**: Automatically classifies errors into specific categories
- **System Context Collection**: Captures detailed system metrics during error conditions
- **Hardware Status Reporting**: Monitors for overheating, memory pressure, and throttling
- **Error Frequency Analysis**: Tracks error patterns to identify recurring issues
- **Hardware-Aware Error Reporting**: Includes hardware-specific context with errors
- **Error History**: Maintains a history of errors for trend analysis
- **Integration with Worker Reconnection**: Seamless integration with enhanced worker reconnection system
- **Error Simulation**: Built-in capabilities to simulate errors for testing
- **Error Reporting Metrics**: Dashboard integration for monitoring error patterns

## Key Components

### 1. EnhancedErrorReporter

The `EnhancedErrorReporter` class provides comprehensive error reporting with detailed system context:

- Error categorization based on message patterns
- System context collection (CPU, memory, disk, network)
- Hardware status detection (overheating, memory pressure, throttling)
- Error frequency analysis and pattern detection
- Error history management

### 2. Worker Error Reporting Integration

The error reporting system can be integrated with:

- Standard Worker Client: Using the `integrate_error_reporting` function
- Enhanced Worker Client: Through the `EnhancedWorkerClientWithErrorReporting` class

### 3. Error Categories

Errors are categorized into standardized types through the `ErrorCategory` enum:

| Category | Description | Examples |
|----------|-------------|----------|
| `RESOURCE_ALLOCATION_ERROR` | Failures related to resource allocation | Out of memory, disk full |
| `RESOURCE_CLEANUP_ERROR` | Failures related to resource cleanup | Unable to free resources |
| `NETWORK_CONNECTION_ERROR` | Network connectivity issues | Connection refused, reset |
| `NETWORK_TIMEOUT_ERROR` | Network timeouts | Request timed out |
| `HARDWARE_AVAILABILITY_ERROR` | Hardware unavailability | No GPU found |
| `HARDWARE_CAPABILITY_ERROR` | Hardware capability mismatches | Insufficient compute capability |
| `HARDWARE_PERFORMANCE_ERROR` | Hardware performance degradation | Thermal throttling |
| `WORKER_DISCONNECTION_ERROR` | Worker node disconnections | Connection closed |
| `WORKER_CRASH_ERROR` | Worker node crashes | Segmentation fault |
| `WORKER_OVERLOAD_ERROR` | Worker node overload conditions | Too many requests |
| `TEST_EXECUTION_ERROR` | Test execution failures | Assertion failed |
| `TEST_DEPENDENCY_ERROR` | Test dependency issues | Module not found |
| `TEST_CONFIGURATION_ERROR` | Test configuration problems | Invalid configuration |
| `UNKNOWN_ERROR` | Unknown or uncategorized errors | - |

### 4. System Context

The system context includes:

- **Timestamp**: When the error occurred
- **Hostname**: Worker node hostname
- **Platform**: Operating system and version
- **Architecture**: CPU architecture
- **Python Version**: Python interpreter version
- **Hardware Type**: Primary hardware type being used

### 5. Hardware Metrics

Hardware metrics include:

- **CPU**: Cores, utilization, frequency
- **Memory**: Total, available, used percentage
- **Disk**: Total, free, used percentage
- **Network**: Bytes sent/received, packets sent/received, errors
- **GPU (if available)**: Name, memory usage, compute capability

### 6. Hardware Status

The system checks for:

- **Overheating**: CPU or GPU temperature above 85°C
- **Memory Pressure**: System or GPU memory usage above 90%
- **Throttling**: CPU frequency significantly lower than maximum

### 7. Error Frequency Analysis

Error frequency analysis includes:

- **Same Type Count**: Number of errors of the same type in the last 1h, 6h, 24h
- **Similar Message Count**: Number of errors with similar messages in the last 1h, 6h, 24h
- **Total Errors**: Total number of errors in history
- **Recurring Flag**: Whether the error is recurring (more than 2 similar errors in the last hour)

## Usage Guide

### 1. Running the Enhanced Worker with Error Reporting

```bash
# Start worker with basic error reporting
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY

# Enable error simulation for testing
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --simulate-error \
  --error-type "ResourceAllocationError" \
  --error-message "Simulated out of memory error for testing error reporting"

# Configure error history size
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --error-history-size 200

# Enable verbose logging
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --log-level DEBUG
```

### 2. Integrating Error Reporting with Existing Workers

You can integrate error reporting with existing worker clients:

```python
from duckdb_api.distributed_testing.worker import WorkerClient
from duckdb_api.distributed_testing.worker_error_reporting import integrate_error_reporting

# Create standard worker client
worker = WorkerClient(
    coordinator_url="ws://localhost:8080",
    api_key="YOUR_API_KEY"
)

# Integrate enhanced error reporting
worker = integrate_error_reporting(worker)

# Worker now has enhanced error reporting capabilities
# All errors reported by this worker will include detailed context
```

### 3. Using the Enhanced Error Reporter Directly

You can use the enhanced error reporter directly in your code:

```python
from duckdb_api.distributed_testing.worker_error_reporting import EnhancedErrorReporter

# Create error reporter
error_reporter = EnhancedErrorReporter(
    worker_id="worker-123",
    capabilities={"hardware_types": ["cuda", "cpu"]}
)

try:
    # Your code that might raise an exception
    do_something_risky()
except Exception as e:
    # Create enhanced error report
    enhanced_error = error_reporter.create_enhanced_error_report(
        error_type="TaskExecutionError",
        message=str(e),
        task_id="task-456"
    )
    
    # Report error with enhanced context
    report_error_to_coordinator(enhanced_error)
```

### 4. Examining Error Reports

Enhanced error reports include:

```json
{
  "type": "TaskExecutionError",
  "message": "Out of memory",
  "traceback": "...",
  "timestamp": "2025-03-18T14:30:45.123456",
  "worker_id": "worker-123",
  "task_id": "task-456",
  "error_category": "RESOURCE_ALLOCATION_ERROR",
  "system_context": {
    "timestamp": "2025-03-18T14:30:45.123456",
    "hostname": "worker-node-1",
    "platform": "Linux",
    "platform_version": "5.15.0-generic",
    "architecture": "x86_64",
    "python_version": "3.10.6",
    "hardware_type": "cuda",
    "metrics": {
      "cpu": {
        "percent": 85.2,
        "count": 16,
        "physical_count": 8,
        "frequency_mhz": 3600.0
      },
      "memory": {
        "total_gb": 32.0,
        "available_gb": 1.2,
        "used_percent": 96.3
      },
      "disk": {
        "total_gb": 500.0,
        "free_gb": 250.0,
        "used_percent": 50.0
      },
      "network": {
        "bytes_sent": 12345678,
        "bytes_recv": 87654321,
        "packets_sent": 12345,
        "packets_recv": 54321,
        "error_in": 0,
        "error_out": 0,
        "drop_in": 0,
        "drop_out": 0
      }
    },
    "gpu_metrics": {
      "count": 2,
      "devices": [
        {
          "name": "NVIDIA RTX 4090",
          "index": 0,
          "compute_capability": "8.9",
          "total_memory_gb": 24.0,
          "free_memory_gb": 0.5,
          "used_memory_gb": 23.5,
          "memory_utilization": 97.9
        },
        {
          "name": "NVIDIA RTX 4090",
          "index": 1,
          "compute_capability": "8.9",
          "total_memory_gb": 24.0,
          "free_memory_gb": 4.5,
          "used_memory_gb": 19.5,
          "memory_utilization": 81.3
        }
      ]
    }
  },
  "hardware_context": {
    "hardware_type": "cuda",
    "hardware_types": ["cuda", "cpu"],
    "hardware_status": {
      "overheating": false,
      "memory_pressure": true,
      "throttling": false
    }
  },
  "error_frequency": {
    "same_type": {
      "last_1h": 3,
      "last_6h": 5,
      "last_24h": 8
    },
    "similar_message": {
      "last_1h": 3,
      "last_6h": 5,
      "last_24h": 8
    },
    "total_errors": 28,
    "recurring": true
  }
}
```

## Integration with Coordinator Error Handling

The enhanced worker error reporting system complements the coordinator's enhanced error handling system:

1. **Worker**: Captures detailed error context including system metrics and hardware status
2. **Coordinator**: Receives enhanced error reports and uses them to make intelligent recovery decisions

For a complete end-to-end error handling pipeline:

1. Run workers with enhanced error reporting enabled
2. Run coordinator with enhanced error handling enabled
3. Configure both coordinator and workers to use the same error categories
4. Enable the coordinator's error pattern detection to identify systemic issues

## Advanced Features

### 1. Error Simulation

You can simulate errors to test the error reporting and recovery system:

```bash
# Simulate a resource allocation error
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --simulate-error \
  --error-type "ResourceAllocationError" \
  --error-message "Simulated out of memory error"

# Simulate a network error
python run_enhanced_worker_with_error_reporting.py \
  --coordinator-host localhost \
  --coordinator-port 8765 \
  --api-key YOUR_API_KEY \
  --simulate-error \
  --error-type "NetworkError" \
  --error-message "Connection reset by peer"
```

### 2. Error Metrics Dashboard

When used with the monitoring dashboard, error metrics are available:

- Error frequency by category
- Error trends over time
- System metrics at the time of errors
- Hardware status correlation with errors
- Worker node health metrics

To enable error metrics in the dashboard:

```bash
python run_monitoring_dashboard.py --error-metrics
```

### 3. Custom Error Categories

You can extend the error categories to include domain-specific error types:

```python
from duckdb_api.distributed_testing.distributed_error_handler import ErrorCategory
from enum import Enum

# Create custom error categories
class CustomErrorCategory(Enum):
    MODEL_LOADING_ERROR = "model_loading_error"
    QUANTIZATION_ERROR = "quantization_error"
    DATA_PREPROCESSING_ERROR = "data_preprocessing_error"

# Use custom categories in error reporter
error_reporter = EnhancedErrorReporter(
    worker_id="worker-123",
    capabilities={"hardware_types": ["cuda", "cpu"]},
    custom_categories=CustomErrorCategory
)
```

## Conclusion

The Enhanced Worker Error Reporting system provides comprehensive error context, categorization, and system metrics to help coordinators make more intelligent recovery decisions. By capturing detailed information about errors, including system state and hardware status, it enables more effective diagnosis and recovery from errors in the distributed testing framework.

When used in conjunction with the Enhanced Error Handling system on the coordinator side, it creates a powerful end-to-end error handling pipeline that can significantly improve the reliability and resilience of the distributed testing framework.