# Advanced Browser Recovery Strategies

**Implementation Date:** March 19, 2025

## Overview

The Advanced Browser Recovery Strategies module enhances the Circuit Breaker pattern with sophisticated, model-aware recovery approaches tailored to different browser types. This system provides progressive recovery that intelligently escalates from simple retries to browser-specific optimizations, cross-browser fallbacks, and simulation mode as a last resort.

## Key Features

### 1. Browser-Specific Recovery Strategies

The system offers specialized strategies for each major browser:

- **Chrome**: Performance optimizations for vision models and WebNN support
- **Firefox**: Compute shader optimizations for audio models
- **Edge**: WebNN acceleration for text models
- **Safari**: Limited customization with graceful degradation

### 2. Model-Aware Recovery Approaches

Recovery strategies are optimized for different model types:

- **Text Models**: WebNN acceleration on Edge, shader precompilation, latency optimization
- **Vision Models**: WebGPU optimization, zero-copy buffer transfers, throughput focus
- **Audio Models**: Firefox compute shader optimization, workgroup size customization
- **Multimodal Models**: Parallel loading, progressive component loading, memory optimization

### 3. Progressive Recovery with Escalation

Recovery follows a progressive approach with increasing intervention levels:

1. **Minimal**: Simple retries with exponential backoff
2. **Moderate**: Browser restart with preserved arguments
3. **Aggressive**: Browser setting adjustments based on failure type
4. **Fallback**: Switch to different browser optimized for model type
5. **Simulation**: Fall back to simulation mode when hardware fails

### 4. Resource-Aware Recovery

Recovery strategies consider resource constraints:

- Memory optimization for large models
- Batch size reduction for constrained environments
- Progressive loading for multimodal models
- Shared tensor optimization for related models
- Browser-specific resource settings

### 5. Performance Tracking and Analysis

The system includes comprehensive monitoring and analysis:

- Success rate tracking by strategy, browser, and model type
- Execution time measurement for performance optimization
- Historical trend analysis for strategy effectiveness
- Statistical analysis to identify optimal strategies
- Dashboard visualization for recovery metrics

## Usage

### Basic Recovery Usage

```python
from distributed_testing.browser_recovery_strategies import recover_browser

# Recover from a browser failure
async def handle_error(bridge, error):
    success = await recover_browser(bridge, error, {
        "browser": "firefox",
        "model": "whisper-tiny",
        "platform": "webgpu"
    })
    
    if success:
        print("Recovery successful")
    else:
        print("Recovery failed")
```

### Direct Strategy Usage

For more control, you can use the ProgressiveRecoveryManager directly:

```python
from distributed_testing.browser_recovery_strategies import (
    ProgressiveRecoveryManager, BrowserType, ModelType, RecoveryLevel,
    categorize_browser_failure
)

# Create recovery manager
recovery_manager = ProgressiveRecoveryManager()

# Categorize a failure
failure_info = categorize_browser_failure(error, context)

# Execute progressive recovery from a specific level
success = await recovery_manager.execute_progressive_recovery(
    bridge,
    BrowserType.FIREFOX,
    ModelType.AUDIO,
    failure_info,
    start_level=RecoveryLevel.MODERATE  # Skip simple retries
)

# Get performance statistics
stats = recovery_manager.get_strategy_stats()
```

### Integration with Circuit Breaker

This example shows how to integrate with the existing circuit breaker pattern:

```python
from ipfs_accelerate_selenium_bridge import (
    BrowserAutomationBridge, CircuitBreaker, 
    CircuitState, CircuitOpenError
)
from distributed_testing.browser_recovery_strategies import recover_browser

async def run_with_fault_tolerance(model_name, browser_name):
    bridge = BrowserAutomationBridge(
        platform="webgpu",
        browser_name=browser_name,
        model_type=model_name
    )
    
    circuit = CircuitBreaker(f"browser_{browser_name}")
    
    try:
        # Try operation with circuit breaker protection
        result = circuit.execute(
            lambda: bridge.run_test(model_name, "test input")
        )
        return result
    except CircuitOpenError:
        # Circuit is open, cannot retry
        return {"error": "Circuit is open"}
    except Exception as e:
        # Try to recover
        recovered = await recover_browser(bridge, e, {
            "browser": browser_name,
            "model": model_name
        })
        
        if recovered:
            # Retry operation after recovery
            return bridge.run_test(model_name, "test input")
        else:
            # Could not recover
            return {"error": "Recovery failed"}
```

## Browser-Specific Optimizations

The following browser-specific optimizations are applied automatically based on browser type and failure mode:

### Chrome Optimizations

| Failure Type | Optimizations Applied |
|--------------|------------------------|
| Resource Exhaustion | Disable GPU rasterization, GPU vsync, enable low-end device mode |
| GPU Error | Disable GPU process crash limit, disable GPU watchdog |
| Crash | Disable crash reporter, disable breakpad |

### Firefox Optimizations

| Failure Type | Optimizations Applied |
|--------------|------------------------|
| Resource Exhaustion | Reduce memory cache (32MB), limit session history entries |
| GPU Error | Enable unsafe WebGPU operations, disable WebRender |
| Audio Models | Adjust compute shader workgroup size for better performance |
| Crash | Use single process mode, disable multiprocess tabs |

### Edge Optimizations

| Failure Type | Optimizations Applied |
|--------------|------------------------|
| Text Models | Enable WebNN acceleration, enable WebNN compile options |
| API Error | Enable WebNN extension in Dawn |

## Model-Specific Optimizations

Different model types receive specialized optimizations:

### Text Models (BERT, T5, LLAMA, etc.)

- Prefer WebNN on Edge and Chrome
- Conservative batch size (1)
- Optimize for latency over throughput
- Shader precompilation for faster startup

### Vision Models (ViT, CLIP, etc.)

- Prefer WebGPU on Chrome
- Enable zero-copy for images
- GPU memory buffer optimization for video frames
- Higher batch sizes (4)
- Optimize for throughput

### Audio Models (Whisper, Wav2Vec2, etc.)

- Prefer WebGPU on Firefox with compute shaders
- Firefox-specific compute shader optimizations
- Custom workgroup sizes for compute shaders

### Multimodal Models (CLIP, LLaVA, etc.)

- Enable parallel loading for model components
- Progressive loading for memory efficiency
- Tensor sharing across components
- Conservative batch size (1)
- Memory-focused optimizations

## Recovery Strategy Selection

The system intelligently selects strategies based on failure type:

| Failure Type | Starting Recovery Level |
|--------------|-------------------------|
| Launch Failure | Moderate (Browser Restart) |
| Crash | Moderate (Browser Restart) |
| API Error | Aggressive (Settings Adjustment) |
| GPU Error | Aggressive (Settings Adjustment) |
| Other | Minimal (Simple Retry) |

## Browser Fallback Order

The system uses an optimal fallback order based on model type:

### Text Models
1. Edge (best WebNN support)
2. Chrome
3. Firefox

### Audio Models
1. Firefox (best compute shader support)
2. Chrome
3. Edge

### Vision Models
1. Chrome (best WebGPU support)
2. Edge
3. Firefox

### Multimodal Models
1. Chrome
2. Edge
3. Firefox

## Dashboard and Visualization

The recovery system integrates with the dashboard visualization to display:

- Success rates by browser, model type, and strategy
- Execution times for different recovery approaches
- Strategy history and effectiveness over time
- Browser and model-specific optimizations applied
- Progressive recovery paths and outcomes

## System Architecture

### Key Classes

- **BrowserRecoveryStrategy**: Base class for all recovery strategies
- **SimpleRetryStrategy**: Implements retry with exponential backoff
- **BrowserRestartStrategy**: Manages browser restart with preserved settings
- **SettingsAdjustmentStrategy**: Adjusts browser settings based on failure
- **ModelSpecificRecoveryStrategy**: Applies model-specific optimizations
- **BrowserFallbackStrategy**: Switches to different browser types
- **SimulationFallbackStrategy**: Falls back to simulation mode
- **ProgressiveRecoveryManager**: Orchestrates strategy execution

### Recovery Levels

The system defines five progressive recovery levels:

1. **MINIMAL**: Simple retry, no browser restart
2. **MODERATE**: Browser restart with same settings
3. **AGGRESSIVE**: Browser restart with modified settings
4. **FALLBACK**: Switch to different browser
5. **SIMULATION**: Fall back to simulation mode

### Utility Functions

- **detect_browser_type**: Determines browser type from name
- **detect_model_type**: Identifies model type from name
- **categorize_browser_failure**: Classifies failure types
- **recover_browser**: High-level recovery function

## Future Enhancements

Planned enhancements to the recovery system:

1. **Machine Learning-Based Strategy Selection**: Using past performance to predict optimal recovery paths
2. **Advanced Failure Fingerprinting**: More detailed failure categorization using stack traces and context
3. **Cross-Node Recovery Coordination**: Coordinated recovery across distributed nodes
4. **Resource Monitoring Integration**: Proactive recovery based on resource trends
5. **Platform-Specific Hardware Optimizations**: Enhanced hardware support for Windows, Linux, macOS
6. **Container-Based Browser Isolation**: Improved reliability through containerization

## Best Practices

1. **Match Browser to Model**: Use Edge for text models, Firefox for audio models, Chrome for vision models
2. **Enable Compute Shaders for Audio**: Always enable compute shaders for audio models on Firefox
3. **Use Progressive Recovery**: Start with simpler strategies before more invasive approaches
4. **Monitor Strategy Performance**: Review strategy stats to identify optimal approaches
5. **Customize for Environment**: Adjust recovery timeouts based on system capabilities
6. **Enable Simulation Fallback**: Always allow simulation fallback as a last resort
7. **Integrate with Circuit Breaker**: Use recovery alongside circuit breaker for comprehensive protection
8. **Collect Performance Metrics**: Track recovery performance for continuous improvement

## Troubleshooting

### Common Issues

- **Repeated Strategy Failures**: Check browser installation and verify hardware acceleration
- **Slow Recovery Times**: Adjust timeouts and consider environment constraints
- **Simulation Always Used**: Ensure hardware acceleration is available and enabled
- **Browser Launch Failures**: Verify browser executable paths and permissions
- **WebGPU/WebNN Detection Issues**: Update browser to latest version with better support

### Logging and Debugging

Enable detailed logging for troubleshooting:

```bash
export SELENIUM_BRIDGE_LOG_LEVEL=DEBUG
python duckdb_api/distributed_testing/run_fault_tolerance_e2e_test.py --use-real-browsers
```

## Conclusion

The Advanced Browser Recovery Strategies system provides a comprehensive approach to handling browser failures in distributed testing environments. By combining browser-specific, model-aware strategies with progressive recovery, the system ensures maximum resilience and testing effectiveness even in the presence of browser failures and resource constraints.