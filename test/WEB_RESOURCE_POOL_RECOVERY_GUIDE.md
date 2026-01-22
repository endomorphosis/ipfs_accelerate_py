# WebNN/WebGPU Resource Pool Recovery Integration Guide

This document provides comprehensive information about the WebNN/WebGPU Resource Pool integration with the Recovery System and Circuit Breaker Pattern (March 11, 2025), which enhances reliability and fault tolerance when using browser-based hardware acceleration.

## Overview

The WebNN/WebGPU Resource Pool Recovery integration provides a robust, fault-tolerant layer that enables resilient operation with browser-based hardware acceleration. The integration bridges these key components:

1. **ResourcePool**: Central resource management system for model sharing and caching
2. **WebNN/WebGPU Resource Pool**: Manages browser-based hardware acceleration
3. **Recovery System**: Provides fault tolerance with error detection, categorization, and recovery
4. **Circuit Breaker Pattern**: Implements advanced health monitoring and graceful degradation (NEW - March 11, 2025)
5. **Connection Pool Manager**: Manages browser lifecycle with health-aware routing (NEW - March 11, 2025)

This integration enables applications to leverage browser-based WebNN and WebGPU hardware acceleration with automatic error recovery, fallbacks, performance monitoring, and comprehensive health tracking.

## Key Features

- **Automatic Error Recovery**: Detects and recovers from common browser errors
- **Smart Fallbacks**: Intelligently switches between WebNN, WebGPU, and CPU simulation
- **Browser Optimization**: Uses the most appropriate browser for each model type
  - Firefox for audio models (optimized compute shaders)
  - Edge for text models (superior WebNN support)
  - Chrome for vision models (solid WebGPU support)
- **Performance Monitoring**: Tracks browser and platform health for smart decisions
- **Graceful Degradation**: Falls back to simulation when hardware is unavailable
- **Comprehensive Metrics**: Provides detailed statistics on recovery and performance
- **Seamless Integration**: Works transparently through the ResourcePool interface

### New Features (March 11, 2025)

- **Circuit Breaker Pattern**: Prevents cascading failures with automatic service isolation
- **Health Scoring (0-100)**: Comprehensive health metrics for each connection
- **Model-Browser Performance Tracking**: Records and leverages historical performance data
- **Intelligent Connection Pooling**: Optimized browser connection lifecycle management
- **Error Categorization and Recovery**: Targeted recovery strategies by error type
- **Browser Health Monitoring**: Real-time monitoring with proactive remediation
- **Performance History-Based Routing**: Routes models based on historical performance data

## Architecture

The integration uses a layered architecture with the new circuit breaker components (March 11, 2025):

```
┌─────────────────────────┐
│      ResourcePool       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ConnectionPoolIntegration │
└───────────┬─────────────┘
            │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ ResourcePoolBridge with │  │  ResourcePoolCircuit-   │
│     Recovery System     │  │     BreakerManager      │
└───────────┬─────────────┘  └───────────┬─────────────┘
            │                            │
            └─────────────┬─────────────┘
                          │
                          ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│    Browser Automation    │  │    ConnectionPool-      │
│    & WebSocket Bridge   │◄─┤       Manager           │
└─────────────────────────┘  └─────────────────────────┘
```

- **ResourcePool**: Provides the main interface for model loading and caching
- **ConnectionPoolIntegration**: Combines connection pooling with circuit breaker pattern (NEW)
- **ResourcePoolBridge with Recovery**: Handles error detection, categorization, and recovery
- **ResourcePoolCircuitBreakerManager**: Implements circuit breaker pattern for health monitoring (NEW)
- **ConnectionPoolManager**: Manages browser connections with lifecycle tracking (NEW)
- **Browser Automation & WebSocket**: Manages browser instances and communication

## Usage

### Loading Models with WebNN/WebGPU Acceleration

```python
from resource_pool import global_resource_pool

# For WebGPU acceleration
model = global_resource_pool.get_model(
    model_type="text",
    model_name="bert-base-uncased",
    constructor=my_constructor_function,
    hardware_preferences={
        "priority_list": ["webgpu", "cpu"],
        "browser": "chrome"
    }
)

# For WebNN acceleration
model = global_resource_pool.get_model(
    model_type="text",
    model_name="bert-base-uncased",
    constructor=my_constructor_function,
    hardware_preferences={
        "priority_list": ["webnn", "cpu"],
        "browser": "edge"
    }
)
```

### Hardware Preferences

You can specify various hardware preferences:

```python
hardware_preferences = {
    # Prioritize hardware platforms in order
    "priority_list": ["webgpu", "webnn", "cpu"],
    
    # Specify browser (firefox optimized for audio models)
    "browser": "firefox",
    
    # Set precision for quantization (16, 8, 4)
    "precision": 8,
    
    # Enable mixed precision for better performance
    "mixed_precision": True,
    
    # Enable optimizations
    "compute_shaders": True,  # Especially for audio models
    "precompile_shaders": True,  # For faster startup
    "parallel_loading": True,  # For multimodal models
}
```

### Concurrent Execution

```python
# Get models
model1 = global_resource_pool.get_model("text", "bert-base-uncased", ...)
model2 = global_resource_pool.get_model("vision", "vit-base-patch16-224", ...)

# Prepare inputs
models_and_inputs = [
    (model1, {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}),
    (model2, {"pixel_values": [...]})
]

# Execute concurrently with automatic WebNN/WebGPU acceleration
results = global_resource_pool.execute_concurrent(models_and_inputs)
```

### Getting Statistics and Metrics

```python
# Get comprehensive stats including recovery metrics
stats = global_resource_pool.get_stats()

# Access WebNN/WebGPU stats
web_stats = stats["web_resource_pool"]
print(f"WebNN/WebGPU available: {web_stats['available']}")
print(f"WebNN/WebGPU initialized: {web_stats['initialized']}")

# Access recovery statistics if available
if "recovery_stats" in web_stats:
    recovery_stats = web_stats["recovery_stats"]
    print(f"Total recovery attempts: {recovery_stats['total_recovery_attempts']}")
    print(f"Error categories: {recovery_stats['error_categories']}")
```

## Error Categories and Recovery Strategies

The recovery system categorizes errors and applies appropriate strategies with enhanced circuit breaker pattern (March 11, 2025):

| Error Category | Description | Recovery Strategies | Circuit Breaker Action |
|----------------|-------------|---------------------|------------------------|
| CONNECTION | WebSocket or browser connection issues | Retry, restart browser, try another browser | Open circuit after 5 consecutive failures |
| BROWSER_CRASH | Browser process crashed | Restart browser, try another browser | Open circuit immediately |
| OUT_OF_MEMORY | Out of memory errors | Reduce model size, reduce precision, try another browser | Trigger ultra-low precision, open circuit if persists |
| TIMEOUT | Operation timed out | Retry with delay, restart browser | Progressive backoff with ping check |
| UNSUPPORTED_OPERATION | Operation not supported on platform | Try another platform (WebNN/WebGPU/CPU), try another browser | Route to alternative browser |
| BROWSER_CAPABILITY | Browser lacks required capability | Try another browser, try another platform | Remove from routing options |
| MODEL_INCOMPATIBLE | Model not compatible with backend | Try another platform, reduce precision, reduce model size | Record in model-browser performance data |
| WEBSOCKET | WebSocket communication error | Reconnect with progressive backoff | Half-open state after timeout |
| RESOURCE | Memory/CPU resource issues | Browser restart, suggest ultra-low precision | Open circuit with memory threshold alert |
| INITIALIZATION | Setup error during initialization | Reconnection, browser restart | Remove from initial connection pool |

## Environment Variables

Configure behavior with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| INIT_WEB_RESOURCE_POOL | 1 | Auto-initialize WebNN/WebGPU resource pool |
| FORCE_WEB_RESOURCE_POOL | 0 | Force use of WebNN/WebGPU resource pool |
| RESOURCE_POOL_LOW_MEMORY | 0 | Enable low memory mode |

## Browser-Specific Optimizations

Different browsers excel at different model types:

| Browser | Best For | Optimization | Improvement |
|---------|----------|--------------|-------------|
| Firefox | Audio models | Compute shader optimization | 20-25% faster for Whisper, CLAP |
| Edge | Text models | WebNN optimization | More efficient for BERT, T5 models |
| Chrome | Vision models | Shader precompilation | Better for ViT, CLIP models |

## Integration with Existing Code

The integration is designed to work transparently with existing code that uses the ResourcePool:

```python
# Existing code remains unchanged
model = global_resource_pool.get_model("text", "bert-base-uncased", constructor)

# Add hardware preferences to enable WebNN/WebGPU with recovery
model = global_resource_pool.get_model(
    "text", 
    "bert-base-uncased", 
    constructor,
    hardware_preferences={"priority_list": ["webgpu", "cpu"]}
)
```

## Recovery Process

When an error occurs during model loading or inference, the recovery system:

1. **Categorizes the error** into one of the error categories
2. **Selects a recovery strategy** based on the error category and context
3. **Applies the strategy** (e.g., restart browser, switch browser, reduce precision)
4. **Retries the operation** with the modified context
5. **Falls back** to simulation mode if all recovery attempts fail

## Implementation Details

The integration consists of these key components:

1. **ResourcePoolBridgeRecovery**: Provides error categorization and recovery strategies
2. **ResourcePoolBridgeWithRecovery**: Wraps the base bridge with recovery capabilities
3. **ResourcePoolBridgeIntegrationWithRecovery**: Integrates the recovery system with the resource pool

### Circuit Breaker Pattern Implementation (March 11, 2025)

The new circuit breaker pattern consists of these components:

1. **ResourcePoolCircuitBreaker**: Core implementation of the circuit breaker with three states:
   - **CLOSED**: Normal operation - requests flow through
   - **OPEN**: Circuit is open - fast fail for all requests
   - **HALF_OPEN**: Testing if service has recovered - limited requests

2. **BrowserHealthMetrics**: Tracks and analyzes browser connection health:
   - Response times tracking
   - Error rates and patterns
   - Resource usage monitoring
   - Connection stability metrics
   - Model-specific performance data

3. **ConnectionPoolManager**: Manages browser connections with lifecycle tracking:
   - Intelligent connection allocation
   - Browser-specific optimizations
   - Dynamic scaling based on workload
   - Health-aware routing decisions

4. **ConnectionPoolIntegration**: Combines connection pooling with circuit breaker:
   - Health monitoring integration
   - Model-browser performance tracking
   - Error recovery coordination
   - Health score-based routing

### Health Scoring System

The circuit breaker implements a sophisticated health scoring system (0-100) based on multiple factors:

```python
health_score = weighted_average([
    error_rate_factor,      # Heavily penalizes high error rates
    response_time_factor,   # Penalizes slow response times
    consecutive_failures,   # Tracks failure patterns
    connection_drops,       # Monitors connection stability
    memory_usage,           # Tracks resource constraints
    ping_latency            # Measures communication efficiency
])
```

## Advanced Features

### Cross-Model Tensor Sharing

The integration supports cross-model tensor sharing for improved efficiency:

```python
# Setup tensor sharing with memory limit
manager = global_resource_pool.web_resource_pool.setup_tensor_sharing(max_memory_mb=2048)

# Models using the same resource pool can share tensors
model1 = global_resource_pool.get_model("text", "bert-base-uncased", ...)
model2 = global_resource_pool.get_model("text", "t5-small", ...)

# Tensors will be automatically shared when possible
```

### Health Monitoring with Circuit Breaker Pattern

The integration includes comprehensive health monitoring with circuit breaker pattern for browsers and platforms (March 11, 2025):

```python
# Get health status with circuit breaker metrics
health = global_resource_pool.web_resource_pool.get_health_status()

# Access comprehensive health metrics
if 'circuit_breaker' in health:
    circuit_metrics = health['circuit_breaker']
    print(f"Overall health score: {circuit_metrics['overall_health_score']}")
    print(f"Open circuits: {circuit_metrics['open_circuit_count']}")
    print(f"Half-open circuits: {circuit_metrics['half_open_circuit_count']}")
    
    # Access connection health scores
    for conn_id, conn_health in circuit_metrics['connections'].items():
        print(f"Connection {conn_id} health: {conn_health['health_score']}/100")
        print(f"  Browser: {conn_health['browser']}")
        print(f"  State: {conn_health['circuit_state']}")
        print(f"  Error rate: {conn_health['error_rate']*100:.1f}%")
        
    # Access browser recommendations
    if 'browser_recommendations' in health:
        for model_type, browser in health['browser_recommendations'].items():
            print(f"Recommended browser for {model_type}: {browser}")
```

## Troubleshooting

### Common Issues

1. **Browser Connection Failures**
   - Ensure that the browser (Chrome, Firefox, Edge) is installed and working
   - Check if Selenium WebDriver is installed and properly configured

2. **Model Loading Errors**
   - Verify that the model is compatible with WebNN/WebGPU
   - Try with a smaller model or reduced precision

3. **Slow Performance**
   - Check browser health metrics
   - Consider using browser-specific optimizations

### Logs and Diagnostics

Enable detailed logging for diagnostics:

```python
import logging
logging.getLogger("ResourcePool").setLevel(logging.DEBUG)
```

## Circuit Breaker Pattern Overview

The new circuit breaker pattern (March 11, 2025) works as follows:

### States and Transitions

1. **CLOSED State** (Normal Operation)
   - All requests are allowed to flow through normally
   - Failures are tracked and counted
   - When consecutive failures reach the threshold (default: 5), circuit transitions to OPEN

2. **OPEN State** (Preventing Cascading Failures)
   - All requests are fast-failed without reaching the failing browser
   - After a timeout period (default: 30 seconds), circuit transitions to HALF-OPEN

3. **HALF-OPEN State** (Testing Recovery)
   - Limited requests are allowed through (default: 3 concurrent)
   - Successful requests increment success counter
   - When consecutive successes reach threshold (default: 3), circuit transitions to CLOSED
   - Any failure during HALF-OPEN transitions back to OPEN

### Recovery Process with Circuit Breaker

1. **Error Detection**: A failure is detected and categorized
2. **Circuit Opening**: After consecutive failures, circuit opens
3. **Fast Failing**: Subsequent requests fail fast during timeout
4. **Recovery Testing**: After timeout, limited requests test recovery
5. **Circuit Closing**: Upon successful recovery, circuit closes

### Browser-Specific Optimizations

The circuit breaker enhances browser-specific optimizations by:

1. **Tracking Performance History**: Recording model performance by browser
2. **Browser Recommendation**: Suggesting optimal browsers by model type
3. **Health-Aware Routing**: Routing requests based on browser health scores
4. **Error Pattern Analysis**: Identifying browser-specific error patterns

## Conclusion

The WebNN/WebGPU Resource Pool Recovery integration with Circuit Breaker Pattern (March 11, 2025) provides a robust, fault-tolerant solution for using browser-based hardware acceleration in your applications. By automatically handling errors, monitoring health, providing smart fallbacks, and optimizing for different browsers and model types, it enables reliable and efficient inference with WebNN and WebGPU.

The new circuit breaker pattern significantly enhances reliability by preventing cascading failures, providing intelligent recovery strategies, and enabling health-aware routing decisions. Combined with browser-specific optimizations and model-browser performance tracking, this creates a resilient and efficient system for browser-based AI acceleration.