# WebNN/WebGPU Resource Pool Error Recovery

This document describes the enhanced error recovery mechanisms implemented for the WebNN/WebGPU Resource Pool Bridge.

## Overview

The Resource Pool Error Recovery module provides advanced error handling and recovery capabilities for the WebNN/WebGPU resource pool, improving reliability, stability, and observability of browser-based connections.

Key features:
- **Circuit Breaker Pattern**: Prevent repeated calls to failing services
- **Progressive Recovery Strategies**: Multiple recovery approaches with increasing aggressiveness
- **Model-Specific Error Tracking**: Track errors at both connection and model level
- **Comprehensive Telemetry**: Detailed metrics for monitoring and diagnostics
- **Memory Pressure Management**: Handle high memory situations gracefully

## Why Enhanced Error Recovery is Needed

Browser-based AI acceleration with WebNN and WebGPU introduces unique challenges that make robust error recovery essential:

1. **Browser State Unpredictability**: Browsers can enter inconsistent states due to memory pressure, GPU resets, or background processes.

2. **WebSocket Connection Instability**: The WebSocket bridge between Python and browser can fail due to various factors:
   - Network timeout or congestion
   - Browser process suspension by OS
   - Connection pool exhaustion
   - Message size limits

3. **WebGPU/WebNN Implementation Variance**: Different browsers have varying implementations:
   - Chrome may have different error behavior than Firefox
   - Edge has superior WebNN support but different error patterns
   - Safari has unique WebGPU limitations

4. **Resource Contention**: Multiple models competing for GPU resources can lead to timeout failures or memory allocation errors.

5. **Browser Lifecycle Complexity**: Browsers have complex lifecycles that can cause intermittent failures during:
   - Page refresh
   - Tab switching
   - Device sleep/wake cycles

## Implementation

The `resource_pool_error_recovery.py` module provides a standalone class with static methods that can be integrated into the existing `ResourcePoolBridge` class. The implementation is non-invasive and can be gradually adopted.

### Integration Steps

1. **Import the Module**: Add the error recovery module to your resource pool bridge implementation:

```python
from fixed_web_platform.resource_pool_error_recovery import ResourcePoolErrorRecovery
```

2. **Add Circuit Breaker to BrowserConnection**: Before performing operations like inference, check if the circuit breaker allows it:

```python
# Inside run_inference method
circuit_allowed, circuit_reason = ResourcePoolErrorRecovery.check_circuit_breaker(self, model_id)
if not circuit_allowed:
    logger.warning(f"Circuit breaker prevented inference: {circuit_reason}")
    # Return simulated result with error info
    return {
        'success': True,  # Still return success for compatibility
        'status': 'simulated',
        'model_id': model_id,
        'is_simulation': True,
        'circuit_breaker_active': True,
        'circuit_breaker_reason': circuit_reason,
    }
```

3. **Use Recovery Mechanism**: When a connection is detected as unhealthy, attempt recovery:

```python
# When a connection is detected as unhealthy
if not connection.is_healthy():
    success, method = await ResourcePoolErrorRecovery.recover_connection(connection)
    if success:
        logger.info(f"Successfully recovered connection using {method}")
    else:
        logger.warning(f"Failed to recover connection, will create a new one")
```

4. **Update Circuit Breaker State**: After operations, update the circuit breaker state:

```python
# After inference succeeds or fails
ResourcePoolErrorRecovery.update_circuit_breaker(
    connection, 
    success=operation_succeeded, 
    model_id=model_id, 
    error=error_message if not operation_succeeded else None
)
```

5. **Export Telemetry**: Periodically export telemetry data for monitoring:

```python
# Periodically or on-demand
telemetry = ResourcePoolErrorRecovery.export_telemetry(
    resource_pool=self,
    include_connections=True,  # Include detailed connection data
    include_models=True        # Include detailed model data
)

# Log or save telemetry data
with open('telemetry.json', 'w') as f:
    json.dump(telemetry, f, indent=2)
```

## Architecture

### Circuit Breaker Pattern

The circuit breaker pattern prevents repeated calls to failing services:

- **Closed State**: Normal operation, requests pass through
- **Open State**: After failures exceed threshold, requests fail fast
- **Half-Open State**: After timeout, allows a test request to check if service has recovered

### Recovery Strategies

The recovery mechanism implements multiple strategies with increasing aggressiveness:

1. **Ping Test**: Verify basic connectivity with WebSocket ping
2. **WebSocket Reconnection**: Create a new WebSocket connection
3. **Browser Restart**: Completely restart the browser instance

### Telemetry Export

The telemetry export provides comprehensive data about:

- System resources (CPU, memory)
- Connection health and status
- Circuit breaker states
- Model performance metrics
- Error statistics

## Usage

### Direct Testing

You can test the error recovery tools directly:

```bash
# Test connection recovery
python resource_pool_error_recovery.py --test-recovery --connection-id <connection_id>

# Export telemetry data
python resource_pool_error_recovery.py --export-telemetry --detailed --output telemetry.json
```

### Integration Example

Below is a simple example of integrating these tools into your resource pool bridge:

```python
from fixed_web_platform.resource_pool_error_recovery import ResourcePoolErrorRecovery

class EnhancedResourcePoolBridge(ResourcePoolBridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_enhanced_error_recovery = True
    
    async def run_inference(self, model_id, inputs, retry_attempts=1):
        # Check circuit breaker first
        if self.enable_enhanced_error_recovery and model_id in self.model_connections:
            connection = self.model_connections[model_id]
            circuit_allowed, circuit_reason = ResourcePoolErrorRecovery.check_circuit_breaker(connection, model_id)
            
            if not circuit_allowed:
                logger.warning(f"Circuit breaker prevented inference: {circuit_reason}")
                # Return simulated result
                return {'success': True, 'status': 'simulated', 'circuit_breaker_active': True}
        
        # Normal inference logic
        result = await super().run_inference(model_id, inputs, retry_attempts)
        
        # Update circuit breaker after operation
        if self.enable_enhanced_error_recovery and model_id in self.model_connections:
            connection = self.model_connections[model_id]
            ResourcePoolErrorRecovery.update_circuit_breaker(
                connection,
                success=result.get('success', False),
                model_id=model_id,
                error=result.get('error', None) if not result.get('success', False) else None
            )
        
        return result
```

## Benefits

- **Increased Stability**: Prevent cascading failures with circuit breaker pattern
- **Improved Reliability**: Automatically recover from transient failures
- **Better Observability**: Comprehensive telemetry for debugging
- **Graceful Degradation**: Controlled fallback to simulation
- **Memory Management**: Prevent browser crashes due to memory pressure

## Monitoring

The telemetry data can be integrated with monitoring tools to track:

- Connection health over time
- Error rates by model and connection
- Circuit breaker activations
- Memory usage patterns
- Recovery success rates

## Future Enhancements

Planned future enhancements include:

1. **Predictive Recovery**: Use machine learning to predict failures before they occur
2. **Browser-Specific Strategies**: Optimize recovery strategies by browser type
3. **Distributed Telemetry**: Send telemetry to centralized monitoring system
4. **Automated Testing**: Tools to stress-test and validate recovery mechanisms
5. **Resource Throttling**: Automatically adjust resource usage based on system load