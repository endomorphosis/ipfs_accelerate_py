# Error Handling and Recovery Guide

**Version:** 1.0.0  
**Last Updated:** March 6, 2025

## Overview

This guide explains the comprehensive error handling and recovery system implemented in the WebGPU Streaming Inference framework. The system provides unified error management across all components, with standardized error types, recovery mechanisms, and graceful degradation pathways.

## Core Components

The error handling system consists of these core components:

- **ErrorHandler**: Central class that manages errors and coordinates recovery strategies
- **ErrorTypes**: Standardized error categorization system
- **RecoveryStrategies**: Specialized recovery mechanisms for different error types
- **TelemetrySystem**: Error tracking and aggregation
- **ComponentCallbacks**: Cross-component error propagation and handling

## Key Features

- **Cross-component error propagation**: Errors properly bubble up between components
- **Standardized error categorization**: Consistent error types across all components
- **Graceful degradation**: Progressive fallbacks when errors occur
- **Automatic recovery**: Intelligent recovery strategies for common errors
- **Telemetry collection**: Error tracking data for debugging and optimization
- **Browser-specific handling**: Specialized recovery for different browsers

## ErrorHandler Class

The `ErrorHandler` is the central component of the error handling system:

```javascript
// Basic usage of the ErrorHandler
import { ErrorHandler } from '@ipfs-accelerate/web-platform';

// Create error handler with configuration
const errorHandler = new ErrorHandler({
  mode: "graceful",  // or "strict"
  reportErrors: true,
  autoRecovery: true,
  maxRetries: 3
});

// Handle an error with context
try {
  // Some operation that might fail
} catch (error) {
  const result = errorHandler.handleError(
    error,
    {
      component: "webgpu_inference",
      operation: "token_generation",
      modelType: "text"
    },
    true  // Is recoverable
  );
  
  if (result.recovered) {
    console.log("Successfully recovered from error");
    // Continue with result.recoveryResult
  } else {
    console.error("Failed to recover:", result.error);
    // Implement fallback strategy
  }
}
```

### Constructor Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `mode` | string | Error handling mode ("graceful" or "strict") | "graceful" |
| `reportErrors` | boolean | Whether to report errors to telemetry | true |
| `autoRecovery` | boolean | Whether to attempt automatic recovery | true |
| `maxRetries` | number | Maximum number of recovery attempts | 3 |

### Methods

#### `handleError(error, context, recoverable)`

Handle an error with appropriate recovery strategies.

**Parameters:**
- `error` (Error): The error that occurred
- `context` (object): Context information about the error
- `recoverable` (boolean): Whether the error is recoverable

**Returns:**
- `object`: Error handling result with recovery information

#### `registerErrorCallback(errorType, callback)`

Register a callback for a specific error type.

**Parameters:**
- `errorType` (string): Type of error
- `callback` (function): Callback function

**Returns:**
- None

#### `setRecoveryStrategy(errorType, strategyFunction)`

Set a custom recovery strategy for an error type.

**Parameters:**
- `errorType` (string): Type of error
- `strategyFunction` (function): Recovery function

**Returns:**
- None

## Standardized Error Types

The framework uses standardized error types across all components:

| Error Type | Description | Example Scenarios | Recoverable |
|------------|-------------|-------------------|-------------|
| `WebGPUUnsupportedError` | WebGPU not supported | Unsupported browser, old hardware | Yes |
| `MemoryError` | Out of memory error | Large models, memory leaks | Yes |
| `TimeoutError` | Operation timed out | Slow device, complex model | Yes |
| `ShaderCompilationError` | Shader compilation failed | Unsupported features, bugs | Sometimes |
| `ResourceExhaustedError` | Resource limits exceeded | Too many models, large workload | Yes |
| `ModelLoadingError` | Failed to load model | Network issues, corrupt model | Yes |
| `InvalidConfigurationError` | Invalid configuration | Bad parameters, incompatible settings | Yes |
| `RuntimeExecutionError` | Error during inference | Numerical errors, bad inputs | Sometimes |
| `BrowserSpecificError` | Browser-specific issue | Safari WebGPU limitations | Sometimes |
| `ConnectionError` | WebSocket connection issue | Network problems, server down | Yes |

## Example: Error Handling Implementation

```javascript
import { WebPlatformAccelerator, ErrorHandler } from '@ipfs-accelerate/web-platform';

// Create custom error handler
const errorHandler = new ErrorHandler({
  mode: "graceful",
  reportErrors: true,
  autoRecovery: true,
  maxRetries: 3
});

// Register custom recovery strategies
errorHandler.setRecoveryStrategy("MemoryError", async (context) => {
  console.log("Recovering from memory error...");
  
  // Implement memory recovery strategy
  if (context.component === "webgpu_inference") {
    // Clear memory caches
    await clearMemoryCaches();
    
    // Reduce model precision if possible
    if (context.config && context.config.quantization === "int4") {
      return {
        ...context,
        config: { 
          ...context.config,
          quantization: "int8",  // Reduce precision
          maxBatchSize: Math.max(1, context.config.maxBatchSize / 2)  // Reduce batch size
        }
      };
    }
    
    // Last resort: load smaller model
    if (context.modelSize === "large") {
      return {
        ...context,
        modelSize: "medium",
        reloadModel: true
      };
    }
  }
  
  return null;  // Can't recover
});

// Create accelerator with custom error handler
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  errorHandler: errorHandler,
  config: {
    quantization: "int4",
    adaptiveBatchSize: true
  }
});

// Register component-specific handlers
accelerator.on('error', (error) => {
  console.error("Accelerator error:", error);
});

// Use the accelerator with error handling
try {
  const endpoint = accelerator.createEndpoint();
  
  await endpoint({
    text: "Example prompt",
    callback: (token) => console.log(token)
  });
} catch (error) {
  console.error("Unrecoverable error:", error);
  // Implement fallback to simpler model or CPU inference
}
```

## Browser-Specific Error Handling

Different browsers require specialized error handling approaches:

### Chrome/Edge

```javascript
// Chrome/Edge error handling configuration
const chromeErrorConfig = {
  memoryManagement: "aggressive",
  shaderRecovery: true,
  asyncErrorHandling: true,
  trackedResources: ["buffers", "textures", "adapters"],
  diagnosticMode: false
};

// Register Chrome/Edge specific handlers
if (browserInfo.name.toLowerCase() === "chrome" || browserInfo.name.toLowerCase() === "edge") {
  errorHandler.registerErrorCallback("ShaderCompilationError", async (error) => {
    console.warn("Chrome shader compilation error:", error);
    
    // Attempt shader simplification
    if (error.context && error.context.shader) {
      const simplifiedShader = await simplifyShader(error.context.shader);
      if (simplifiedShader) {
        return { ...error.context, shader: simplifiedShader };
      }
    }
    
    return null;
  });
}
```

### Firefox

```javascript
// Firefox error handling configuration
const firefoxErrorConfig = {
  memoryManagement: "standard",
  shaderRecovery: false,  // Firefox has better built-in recovery
  jitErrorHandling: true,
  diagnosticMode: false
};

// Register Firefox specific handlers
if (browserInfo.name.toLowerCase() === "firefox") {
  errorHandler.registerErrorCallback("WebGPUCrashError", async (error) => {
    console.warn("Firefox WebGPU crash detected:", error);
    
    // Firefox-specific: Recreate context and reload model
    await recreateWebGPUContext();
    return { reloadModel: true };
  });
}
```

### Safari

```javascript
// Safari error handling configuration
const safariErrorConfig = {
  memoryManagement: "very_aggressive",
  metalErrorHandling: true,
  timeoutMultiplier: 2.0,  // Longer timeouts for Safari
  diagnosticMode: true
};

// Register Safari specific handlers
if (browserInfo.name.toLowerCase() === "safari") {
  // Memory pressure handler (critical for Safari)
  errorHandler.registerErrorCallback("MemoryError", async (error) => {
    console.warn("Safari memory pressure detected:", error);
    
    // Safari-specific memory recovery
    const recoverySteps = [
      { action: "reduce_batch_size", new_size: 1 },
      { action: "increase_precision", new_precision: "int8" },
      { action: "reduce_context_length", new_length: 512 },
      { action: "disable_features", features: ["kv_cache", "compute_transfer_overlap"] },
      { action: "reload_model", size: "small" }
    ];
    
    // Try each recovery step until one works
    for (const step of recoverySteps) {
      const result = await applySafariRecoveryStep(step);
      if (result.success) {
        return result.context;
      }
    }
    
    return null;  // Can't recover
  });
  
  // Safari timeout handler
  errorHandler.registerErrorCallback("TimeoutError", async (error) => {
    console.warn("Safari timeout detected:", error);
    
    // Implement progressive timeout extension
    if (error.context && error.context.timeoutMs) {
      const newTimeout = error.context.timeoutMs * 1.5;
      if (newTimeout <= 60000) {  // Cap at 60 seconds
        return { ...error.context, timeoutMs: newTimeout };
      }
    }
    
    return null;
  });
}
```

## Graceful Degradation

The framework implements a progressive fallback system for graceful degradation:

```javascript
// Create degradation path manager
const degradationManager = errorHandler.createDegradationManager({
  paths: [
    {
      name: "webgpu_degradation",
      steps: [
        { feature: "quantization", values: ["int2", "int3", "int4", "int8", "fp16"] },
        { feature: "workgroupSize", values: [[256, 1, 1], [128, 1, 1], [64, 1, 1], [32, 1, 1]] },
        { feature: "computeShaders", values: [true, false] },
        { feature: "platform", values: ["webgpu", "webgl", "cpu"] }
      ]
    },
    {
      name: "model_degradation",
      steps: [
        { feature: "modelSize", values: ["large", "medium", "small", "tiny"] },
        { feature: "contextLength", values: [4096, 2048, 1024, 512, 256] },
        { feature: "featureSet", values: ["full", "reduced", "minimal"] }
      ]
    }
  ],
  onDegradation: (path, step, value) => {
    console.log(`Degrading ${path.name}: ${step.feature} = ${value}`);
  }
});

// Automatically degrade on errors
errorHandler.on('unrecoverable_error', (error) => {
  const newConfig = degradationManager.degradeConfiguration(
    error.context.config,
    error.type
  );
  
  if (newConfig) {
    console.log("Degraded to new configuration:", newConfig);
    reinitializeWithConfig(newConfig);
  } else {
    console.error("Cannot degrade further, falling back to CPU inference");
    fallbackToCPU();
  }
});
```

## Error Telemetry

The framework includes a telemetry system for error tracking and analysis:

```javascript
import { ErrorTelemetry } from '@ipfs-accelerate/web-platform';

// Create error telemetry collector
const telemetry = new ErrorTelemetry({
  enabled: true,
  anonymized: true,
  batchReporting: true,
  maxEntries: 100
});

// Register with error handler
errorHandler.setTelemetryCollector(telemetry);

// View collected telemetry
const errorStats = telemetry.getStatistics();
console.log("Error Statistics:", errorStats);
// Example output:
// {
//   totalErrors: 12,
//   byType: {
//     "MemoryError": 5,
//     "TimeoutError": 3,
//     "ShaderCompilationError": 2,
//     "WebGPUUnsupportedError": 1,
//     "RuntimeExecutionError": 1
//   },
//   byComponent: {
//     "webgpu_inference": 7,
//     "model_loader": 3,
//     "configuration": 2
//   },
//   recoveryRate: 0.75,  // 75% of errors were recovered
//   degradationEvents: 2
// }

// Clear telemetry
telemetry.clear();
```

## WebSocket Error Handling

For WebSocket streaming connections, the framework implements specialized error handling:

```javascript
// Server-side error handling (Node.js example)
wss.on('connection', (ws) => {
  // Set up error handler for WebSocket
  const socketErrorHandler = new WebSocketErrorHandler({
    connection: ws,
    autoReconnect: true,
    maxReconnectAttempts: 3,
    reconnectDelay: 1000,  // 1 second delay
    errorProtocol: standardErrorProtocol
  });
  
  // Handle incoming messages
  ws.on('message', async (message) => {
    try {
      const request = JSON.parse(message);
      
      // Process request
      const response = await processRequest(request);
      ws.send(JSON.stringify(response));
    } catch (error) {
      // Handle error with structured protocol
      socketErrorHandler.handleError(error, {
        requestId: request.id,
        operation: "process_request"
      });
    }
  });
  
  // Handle connection errors
  ws.on('error', (error) => {
    socketErrorHandler.handleConnectionError(error);
  });
});

// Client-side error handling
const socket = new WebSocket('wss://example.com/generate');
const clientErrorHandler = new WebSocketErrorHandler({
  connection: socket,
  autoReconnect: true,
  maxReconnectAttempts: 3
});

// Handle different error responses
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'error') {
    // Handle structured error response
    clientErrorHandler.handleErrorResponse(data);
    
    // Implement recovery based on error type
    if (data.error === 'memory_pressure') {
      console.warn("Memory pressure on server, reducing request size");
      reduceRequestSize();
      retryRequest();
    }
  } else {
    // Process normal response
    processResponse(data);
  }
};
```

## Best Practices

### 1. Use Structured Error Context

Always provide detailed context when handling errors:

```javascript
try {
  // Operation that might fail
} catch (error) {
  errorHandler.handleError(error, {
    component: "model_loader",
    operation: "load_weights",
    modelType: "text",
    modelName: "llama-7b",
    modelPath: modelPath,
    retryCount: currentRetryCount,
    timestamp: Date.now()
  });
}
```

### 2. Implement Component-Specific Handling

Register specialized handlers for different components:

```javascript
// WebGPU component handler
errorHandler.registerErrorCallback("ShaderCompilationError", (error) => {
  if (error.context.component === "webgpu_inference") {
    // Shader-specific recovery for inference component
    return inferenceShaderRecovery(error);
  } else if (error.context.component === "webgpu_preprocessing") {
    // Shader-specific recovery for preprocessing component
    return preprocessingShaderRecovery(error);
  }
  return null;
});

// Model loader component handler
errorHandler.registerErrorCallback("ModelLoadingError", (error) => {
  if (error.context.operation === "download_weights") {
    // Network recovery
    return networkRecovery(error);
  } else if (error.context.operation === "parse_model") {
    // Parse recovery
    return parseRecovery(error);
  }
  return null;
});
```

### 3. Implement Progressive Recovery

Try multiple recovery strategies with increasing impact:

```javascript
function memoryErrorRecovery(error) {
  // Try strategies in order of increasing impact
  const strategies = [
    clearUnusedCaches,          // Least impact
    reduceActivationStorage,
    reduceWorkingSet,
    reduceModelPrecision,
    reduceContextLength,
    unloadNonEssentialLayers,
    reloadSmallerModel          // Most impact
  ];
  
  for (const strategy of strategies) {
    const result = strategy(error.context);
    if (result) {
      return result;
    }
  }
  
  return null;  // Could not recover
}
```

### 4. Handle Browser-Specific Issues

Implement browser-specific recovery strategies:

```javascript
// Detect browser
const browserInfo = detectBrowser();
const browser = browserInfo.name.toLowerCase();

// Register browser-specific handlers
if (browser === "safari") {
  // Safari-specific handling
  errorHandler.registerErrorCallback("TimeoutError", safariTimeoutHandler);
  errorHandler.registerErrorCallback("MemoryError", safariMemoryHandler);
} else if (browser === "firefox") {
  // Firefox-specific handling
  errorHandler.registerErrorCallback("ShaderCompilationError", firefoxShaderHandler);
} else if (browser === "chrome" || browser === "edge") {
  // Chrome/Edge-specific handling
  errorHandler.registerErrorCallback("ResourceExhaustedError", chromiumResourceHandler);
}
```

### 5. Provide User Feedback

Implement user-friendly error messages and recovery UI:

```javascript
errorHandler.on('error', (error) => {
  if (error.recoverable) {
    // Show recoverable error UI
    showRecoverableErrorUI(error.message, error.type);
  } else {
    // Show fatal error UI
    showFatalErrorUI(error.message, error.type);
  }
});

errorHandler.on('recovery_started', (context) => {
  // Show recovery in progress
  showRecoveryProgressUI(context);
});

errorHandler.on('recovery_completed', (result) => {
  // Show recovery success
  hideErrorUI();
  showRecoverySuccessUI(result);
});

errorHandler.on('recovery_failed', (error) => {
  // Show recovery failure
  showRecoveryFailedUI(error);
  suggestAlternatives(error);
});
```

## Advanced Error Patterns

### Memory Leak Detection and Recovery

```javascript
// Memory leak detection and recovery
class MemoryLeakDetector {
  constructor(thresholds) {
    this.baselineMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
    this.thresholds = thresholds || {
      warning: 0.5,   // 50% increase
      critical: 1.0,  // 100% increase
      fatal: 2.0      // 200% increase
    };
    this.checkInterval = null;
    this.gcAttempts = 0;
  }
  
  startMonitoring(interval = 5000) {
    this.checkInterval = setInterval(() => this.checkMemory(), interval);
  }
  
  stopMonitoring() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }
  
  async checkMemory() {
    if (!performance.memory) return;
    
    const currentMemory = performance.memory.usedJSHeapSize;
    const increase = (currentMemory - this.baselineMemory) / this.baselineMemory;
    
    if (increase > this.thresholds.fatal) {
      // Critical memory leak detected
      this.stopMonitoring();
      errorHandler.handleError(new MemoryError("Fatal memory leak detected"), {
        component: "memory_monitor",
        increase: increase,
        currentMemory: currentMemory,
        baselineMemory: this.baselineMemory
      });
    } else if (increase > this.thresholds.critical) {
      // Attempt aggressive memory recovery
      await this.attemptMemoryRecovery("aggressive");
    } else if (increase > this.thresholds.warning) {
      // Attempt standard memory recovery
      await this.attemptMemoryRecovery("standard");
    }
  }
  
  async attemptMemoryRecovery(level) {
    this.gcAttempts++;
    
    if (level === "aggressive" || this.gcAttempts > 3) {
      // Release GPU resources
      await releaseUnusedGPUResources();
      
      // Clear caches
      clearModelCaches();
      
      // Force garbage collection if available
      if (window.gc) window.gc();
      
      // Reset baseline after recovery attempt
      setTimeout(() => {
        if (performance.memory) {
          this.baselineMemory = performance.memory.usedJSHeapSize;
          this.gcAttempts = 0;
        }
      }, 1000);
    } else {
      // Standard recovery - just try to trigger GC
      if (window.gc) window.gc();
    }
  }
}

// Usage
const memLeakDetector = new MemoryLeakDetector();
memLeakDetector.startMonitoring();

// Clean up when done
function cleanup() {
  memLeakDetector.stopMonitoring();
}
```

### WebGPU Device Lost Recovery

```javascript
// WebGPU device lost recovery
class WebGPUDeviceRecovery {
  constructor(accelerator) {
    this.accelerator = accelerator;
    this.device = null;
    this.deviceLostCount = 0;
    this.deviceLostCallbacks = [];
  }
  
  initialize(device) {
    this.device = device;
    
    // Set up device lost handler
    device.lost.then((info) => {
      console.warn("WebGPU device lost:", info.message);
      this.deviceLostCount++;
      
      // Attempt recovery
      this.recoverDevice(info).catch(error => {
        console.error("Failed to recover device:", error);
        this.notifyDeviceLost({
          recoverable: false,
          error: error,
          message: "Failed to recover WebGPU device"
        });
      });
    });
  }
  
  onDeviceLost(callback) {
    this.deviceLostCallbacks.push(callback);
    return this;
  }
  
  notifyDeviceLost(info) {
    for (const callback of this.deviceLostCallbacks) {
      try {
        callback(info);
      } catch (e) {
        console.error("Error in device lost callback:", e);
      }
    }
  }
  
  async recoverDevice(info) {
    if (this.deviceLostCount > 3) {
      // Too many recovery attempts, suggest page reload
      this.notifyDeviceLost({
        recoverable: false,
        exceeded: true,
        message: "Too many device lost events, reload required"
      });
      return;
    }
    
    console.log("Attempting to recover WebGPU device...");
    
    try {
      // Notify about recovery attempt
      this.notifyDeviceLost({
        recoverable: true,
        recovering: true,
        message: "Attempting to recover WebGPU device"
      });
      
      // Wait a moment before recovery
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Re-initialize WebGPU
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("Failed to acquire WebGPU adapter during recovery");
      
      const newDevice = await adapter.requestDevice();
      if (!newDevice) throw new Error("Failed to acquire WebGPU device during recovery");
      
      // Store new device
      this.device = newDevice;
      
      // Re-initialize accelerator with new device
      await this.accelerator.reinitializeWithDevice(newDevice);
      
      // Notify about successful recovery
      this.notifyDeviceLost({
        recoverable: true,
        recovered: true,
        message: "Successfully recovered WebGPU device"
      });
      
      console.log("WebGPU device recovered successfully");
    } catch (error) {
      console.error("Device recovery failed:", error);
      
      // Notify about failed recovery
      this.notifyDeviceLost({
        recoverable: false,
        recovered: false,
        error: error,
        message: "Failed to recover WebGPU device"
      });
      
      throw error;
    }
  }
}

// Usage
const deviceRecovery = new WebGPUDeviceRecovery(accelerator);

// Initialize with WebGPU device
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
deviceRecovery.initialize(device);

// Register callback for device lost events
deviceRecovery.onDeviceLost((info) => {
  if (info.recoverable && info.recovering) {
    showRecoveringUI("Recovering WebGPU context...");
  } else if (info.recoverable && info.recovered) {
    hideRecoveringUI();
    showSuccessMessage("WebGPU context recovered successfully");
  } else {
    showErrorMessage(
      "WebGPU context lost and could not be recovered. Please reload the page.",
      { reload: true }
    );
  }
});
```

## Related Documentation

- [WebGPU Streaming Documentation](WEBGPU_STREAMING_DOCUMENTATION.md)
- [Configuration Validation Guide](CONFIGURATION_VALIDATION_GUIDE.md)
- [Browser-Specific Optimizations](browser_specific_optimizations.md)
- [WebSocket Protocol Specification](websocket_protocol_spec.md)
- [Unified Framework API Reference](unified_framework_api.md)