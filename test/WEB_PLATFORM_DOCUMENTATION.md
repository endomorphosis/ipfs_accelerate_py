# Web Platform Documentation (August 3, 2025) - Updated March 3, 2025

## Overview

This comprehensive documentation covers the web platform implementation for running machine learning models directly in web browsers. The system features extensive optimizations for memory efficiency, cross-browser compatibility, and high performance, allowing models up to 7B parameters to run in browser environments with limited resources. 

**Implementation Status (March 3, 2025):** 75% complete with major components fully functional.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Key Components](#key-components)
3. [Installation & Configuration](#installation--configuration)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
   - [Unified Web Framework (NEW)](#unified-web-framework)
   - [Streaming Inference (NEW)](#streaming-inference)
   - [Ultra-Low Precision](#ultra-low-precision)
   - [Progressive Loading](#progressive-loading)
   - [Browser Adaptation](#browser-adaptation)
   - [WebAssembly Fallback](#webassembly-fallback)
6. [Browser Compatibility](#browser-compatibility)
7. [Performance Optimization](#performance-optimization)
8. [API Reference](#api-reference)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## System Architecture

The web platform implementation follows a modular architecture with clear separation of concerns, designed as a layered system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  ┌────────────┐  │
│  │ Model Tests │  │ Benchmarks  │  │ Interactive Apps    │  │ Dashboard  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  └────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Integration Layer                                 │
│  ┌─────────────────────────┐  ┌───────────────────────────┐  ┌───────────┐  │
│  │ Unified Model API       │  │ Benchmark Database API    │  │ Streaming │  │
│  └─────────────────────────┘  └───────────────────────────┘  │ Pipeline  │  │
│                                                               └───────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Feature Layer                                   │
│  ┌────────────┐ ┌───────────┐ ┌────────────┐ ┌───────────┐ ┌──────────────┐ │
│  │ Browser    │ │ Ultra-Low │ │ WebAssembly│ │Progressive│ │ Device       │ │
│  │ Capability │ │ Precision │ │ Fallback   │ │ Loading   │ │ Adaptation   │ │
│  │ Detector   │ │ Quantizer │ │ System     │ │ System    │ │ System       │ │
│  └────────────┘ └───────────┘ └────────────┘ └───────────┘ └──────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Platform Layer                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ WebGPU Handler│  │ WebNN Handler │  │ Safari Handler  │  │ WebSocket   │ │
│  └───────────────┘  └───────────────┘  └─────────────────┘  │ Handler     │ │
│                                                              └─────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Core Layer                                      │
│  ┌───────────┐  ┌─────────────┐  ┌──────────────────────┐  ┌──────────────┐ │
│  │ Tensor    │  │ Memory      │  │ Shader Management    │  │ Error        │ │
│  │ Operations│  │ Management  │  │ & Compilation        │  │ Handling     │ │
│  └───────────┘  └─────────────┘  └──────────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Browser Capability Detection

The system uses sophisticated browser detection to optimize for each environment:

```python
# Detect browser capabilities
detector = BrowserCapabilityDetector()
capabilities = detector.get_capabilities()
profile = detector.get_optimization_profile()

# Check specific feature support
if detector.get_feature_support("ultra_low_precision"):
    # Enable advanced memory optimization features
```

### 2. Ultra-Low Precision Quantization

Advanced 2-bit and 3-bit quantization provides unprecedented memory efficiency:

```python
# Configure precision settings
precision_config = {
    "embedding": 8,      # 8-bit for embeddings
    "attention.query": 3, # 3-bit for queries
    "attention.key": 3,   # 3-bit for keys
    "feed_forward": 2,   # 2-bit for feed forward
    "lm_head": 4         # 4-bit for output
}

# Apply mixed precision quantization
model = quantize_model_mixed_precision(model, precision_config)
```

### 3. WebAssembly Fallback

Seamless fallback for browsers without WebGPU support:

```python
# Create WebAssembly fallback with SIMD optimization
fallback = WebAssemblyFallback(
    enable_simd=True,
    use_shared_memory=True
)

# Dispatch operation with optimal backend selection
result = dispatch_operation(
    operation="matmul",
    inputs={"a": input_tensor, "b": weight_tensor},
    webgpu_available=detector.get_feature_support("webgpu")
)
```

### 4. Progressive Model Loading

Component-based loading with memory management:

```python
# Create loader with memory optimization
loader = ProgressiveModelLoader(
    model_name="llama-7b", 
    platform="webgpu",
    memory_optimization_level="aggressive",
    prioritize_components=["embeddings", "lm_head", "first_layer"]
)

# Load model with progress reporting
model = await loader.load_async()

# Start inference with partially loaded model
await loader.wait_for_minimum_viable_model()
initial_output = await model.generate("Hello", max_tokens=10)
```

## Installation & Configuration

### Basic Installation

```bash
# Install from npm
npm install ipfs-accelerate-web

# Or use direct script tag
<script src="https://cdn.ipfsaccelerate.io/web/v1.0.0/ipfs-accelerate.min.js"></script>
```

### Configuration

Create a configuration file or use the API to configure:

```javascript
const config = {
  modelsDir: "./models",
  defaultPrecision: "mixed",
  enableWebGPU: true,
  enableWebNN: true,
  enableWasm: true,
  maxMemoryUsageMB: 2048,
  defaultModelSettings: {
    progressiveLoading: true,
    cacheStrategy: "weighted_lru",
    cacheSizeMB: 500
  }
};

// Initialize with configuration
const accelerator = new IPFSAccelerate(config);
```

## Basic Usage

### Loading and Running a Model

```javascript
// Initialize the accelerator
const accelerator = new IPFSAccelerate();

// Load a model with progress reporting
const model = await accelerator.loadModel("llama-2-7b", {
  onProgress: (progress, component, details) => {
    console.log(`Loading ${component}: ${progress * 100}%`);
    console.log(`Memory: ${details.currentMemoryMb}MB, ETA: ${details.estimatedTimeRemainingMs}ms`);
    document.getElementById("progressBar").value = progress * 100;
  },
  onComponentLoaded: (component, stats) => {
    console.log(`Component ${component} loaded in ${stats.loadingTimeMs}ms`);
    if (component === "first_layers") {
      // Enable initial inference as soon as critical components are loaded
      document.getElementById("generateButton").disabled = false;
    }
  }
});

// Wait for minimum viable model to be ready
await model.waitForMinimumViableModel();

// Generate text with standard parameters
const output = await model.generate("Once upon a time", {
  maxTokens: 100,
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
  repetitionPenalty: 1.1
});

console.log(output);
document.getElementById("outputText").textContent = output;
```

### Using Ultra-Low Precision for Maximum Memory Efficiency

```javascript
// Load model with sophisticated ultra-low precision configuration
const model = await accelerator.loadModel("llama-2-7b", {
  precision: {
    mode: "mixed",                    // Use mixed precision across different layers
    defaultBits: 2,                   // Default to 2-bit precision
    criticalLayerBits: {
      "embedding": 8,                 // 8-bit for embeddings (preserve quality)
      "attention.query": 3,           // 3-bit for queries
      "attention.key": 3,             // 3-bit for keys
      "attention.value": 3,           // 3-bit for values
      "feed_forward.up": 2,           // 2-bit for feed forward up-projection
      "feed_forward.down": 2,         // 2-bit for feed forward down-projection
      "layernorm": 8,                 // 8-bit for layer normalization
      "lm_head": 4                    // 4-bit for output projection
    },
    adaptivePrecision: true,          // Dynamically adjust precision
    outlierHandling: "selective_fp16", // Handle outliers with FP16
    calibrationDataset: "default"     // Use default calibration dataset
  },
  
  // Configure progressive loading for faster startup
  loading: {
    progressive: true,
    strategy: "critical_path_first",
    parallelLoading: true,
    minimalModelComponents: ["tokenizer", "embeddings", "first_layers", "lm_head"],
    backgroundLoading: true
  },
  
  // Configure memory management
  memory: {
    limit: navigator.deviceMemory ? navigator.deviceMemory * 1024 : 4096, // Auto-limit based on device
    monitoring: true,
    monitoringIntervalMs: 1000,
    kvCacheStrategy: "sliding_window",
    kvCacheSize: 2048,
    aggressiveComponentUnloading: true
  },
  
  // Configure runtime adaptation
  adaptation: {
    enabled: true,
    adaptToHardware: true,
    adaptToBrowser: true,
    adaptToPowerState: true,
    adaptToNetworkConditions: true
  }
});

// Check the actual effective memory usage
const memoryStats = await model.getMemoryUsage();
console.log(`Total model memory: ${memoryStats.totalMb}MB`);
console.log(`Original model size would be: ${memoryStats.originalSizeMb}MB`);
console.log(`Memory reduction: ${memoryStats.reductionPercentage}%`);
```

### Advanced Loading with Component Control

```javascript
// Create loader with fine-grained component control
const loader = accelerator.createLoader("llama-7b");

// Configure components individually
loader.configureComponents({
  "embeddings": {
    precision: "fp16",
    location: "gpu",
    persistence: "permanent"
  },
  "first_layers": {
    precision: "int8",
    count: 4,
    location: "gpu",
    persistence: "permanent"
  },
  "middle_layers": {
    precision: "int4",
    count: 24,
    location: "gpu",
    persistence: "dynamic",
    swappable: true
  },
  "last_layers": {
    precision: "int8",
    count: 4,
    location: "gpu",
    persistence: "permanent"
  }
});

// Start loading with detailed progress callbacks
const loadingTask = loader.startLoading({
  onProgress: (progress, component, details) => {
    updateProgressUI(component, progress, details);
  },
  onComponentLoaded: (component, stats) => {
    updateComponentStatus(component, "loaded", stats);
  },
  onError: (error, component) => {
    updateComponentStatus(component, "error", error);
    handleComponentError(error, component);
  }
});

// Start inference as soon as critical components are loaded
loader.waitForComponents(["embeddings", "first_layers"]).then(() => {
  // Enable "Quick Start" generation with partial model
  document.getElementById("quickStartButton").disabled = false;
});

// Get fully loaded model when all components are ready
const model = await loadingTask;
document.getElementById("fullModelButton").disabled = false;
```

## Advanced Features

### Unified Web Framework (NEW)

The Unified Web Framework provides a comprehensive integration of all web platform components, offering a single cohesive interface for deploying models to web browsers.

#### Key Features

- **Unified API**: Consistent interface across all browsers and hardware
- **Automatic Detection**: Adapts to available features and capabilities
- **Cross-Browser Optimization**: Browser-specific optimizations
- **Complete Feature Integration**: Brings together all web platform features
- **Comprehensive Metrics**: Detailed performance monitoring
- **Advanced Configuration**: Model-specific configuration profiles

#### Python Interface

```python
from fixed_web_platform.unified_web_framework import (
    WebPlatformAccelerator,
    create_web_endpoint,
    get_optimal_config
)

# Get optimal configuration for a model
config = get_optimal_config("bert-base-uncased", "text")

# Create accelerator with auto-detection
accelerator = WebPlatformAccelerator(
    model_path="bert-base-uncased", 
    model_type="text",
    config=config,
    auto_detect=True
)

# Create inference endpoint
endpoint = accelerator.create_endpoint()

# Run inference
result = endpoint("This is a test of the unified framework")

# Get performance metrics
metrics = accelerator.get_performance_metrics()
print(f"Initialization time: {metrics['initialization_time_ms']:.2f}ms")
print(f"First inference time: {metrics['first_inference_time_ms']:.2f}ms")

# Check which features are in use
feature_usage = accelerator.get_feature_usage()
for feature, used in feature_usage.items():
    print(f"{feature}: {'Enabled' if used else 'Disabled'}")
```

#### Model-Specific Recommendations

The framework applies different optimizations based on model type:

| Model Type | Recommended Configuration | Features |
|------------|---------------------------|----------|
| Embedding (BERT, etc.) | 4-bit, shader precompilation | Fast first inference with good memory efficiency |
| Vision | 4-bit, shader precompilation | Optimized for image processing performance |
| Audio | 8-bit, compute shaders | Best performance on Firefox for audio tasks |
| Multimodal | 4-bit, parallel loading | Efficient loading of multiple components |
| LLMs | 4-bit, streaming inference, KV cache | Memory-efficient text generation |

### Streaming Inference (NEW)

The Streaming Inference system provides token-by-token generation for language models with optimized performance and adaptive batch sizing.

#### Key Features

- **Token-by-Token Generation**: Real-time streaming of generated tokens
- **WebSocket Integration**: Stream results over WebSockets
- **Adaptive Batch Sizing**: Dynamically adjust batch size
- **Low-Latency Optimization**: Minimize latency for interactive applications
- **Memory-Efficient KV Cache**: Optimize memory for long contexts
- **Async/Sync Interfaces**: Both asynchronous and synchronous APIs

#### Python Interface

```python
from fixed_web_platform.webgpu_streaming_inference import (
    WebGPUStreamingInference,
    optimize_for_streaming
)

# Optimize configuration for streaming
config = optimize_for_streaming({
    "quantization": "int4",
    "latency_optimized": True,
    "adaptive_batch_size": True
})

# Create streaming handler
streaming_handler = WebGPUStreamingInference(
    model_path="llama-7b",
    config=config
)

# Option 1: Generate with callback
def token_callback(token, is_last=False):
    print(token, end="", flush=True)
    if is_last:
        print("\nGeneration complete!")

result = streaming_handler.generate(
    "Explain the concept of streaming inference",
    max_tokens=100,
    temperature=0.7,
    callback=token_callback
)

# Option 2: Generate asynchronously
import asyncio

async def generate_async():
    result = await streaming_handler.generate_async(
        "Explain the concept of streaming inference",
        max_tokens=100,
        temperature=0.7
    )
    return result

result = asyncio.run(generate_async())

# Get performance statistics
stats = streaming_handler.get_performance_stats()
print(f"Generated {stats['tokens_generated']} tokens at {stats['tokens_per_second']:.2f} tokens/sec")
print(f"Batch size adaptation: {stats['batch_size_history']}")
```

#### WebSocket Server Example

```python
from fixed_web_platform.webgpu_streaming_inference import start_websocket_server
import asyncio

# Start WebSocket server
asyncio.run(start_websocket_server(
    model_path="llama-7b",
    host="localhost",
    port=8765
))
```

**Client JavaScript**:

```javascript
// Connect to streaming server
const socket = new WebSocket('ws://localhost:8765');

// Send generation request
socket.onopen = () => {
    socket.send(JSON.stringify({
        prompt: "Explain streaming inference",
        max_tokens: 100,
        temperature: 0.7
    }));
};

// Process streaming tokens
socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === "token") {
        document.getElementById("output").textContent += message.token;
    } else if (message.type === "complete") {
        console.log(`Generated ${message.tokens_generated} tokens at ${message.tokens_per_second.toFixed(2)} tokens/sec`);
    }
};
```

#### Adaptive Batch Sizing

The streaming system dynamically adjusts batch size based on performance:

1. Starts with a conservative batch size (1)
2. Monitors token generation performance
3. Increases batch size if performance is good
4. Decreases batch size if performance degrades
5. Maintains optimal throughput automatically

### Advanced Streaming Inference with Real-time Control

```javascript
// Configure streaming options with WebSocket support
const streamingConfig = {
  mode: "token-by-token",
  useWebSocket: true,
  bufferStrategy: "adaptive",
  maxPendingTokens: 128,
  lowLatencyMode: true,
  maxChunkSize: 16,
  streamingEndpoint: "/api/stream",
  errorHandling: "continue"  // Try to continue on minor errors
};

// Set up enhanced streaming text generation
const stream = await model.generateStream(
  "Explain quantum computing in simple terms",
  {
    // Generation parameters
    maxTokens: 1000,
    temperature: 0.7,
    topP: 0.95,
    presencePenalty: 0.1,
    frequencyPenalty: 0.1,
    stopSequences: ["###", "[END]"],
    
    // Control parameters
    streamingConfig: streamingConfig,
    abortController: new AbortController(), // For cancellation
    
    // Callbacks for streaming events
    onTokenReceived: (token, index, timing) => {
      // Update UI with new token
      appendTokenToUI(token);
      
      // Track token generation performance
      logTokenTiming(timing.generationTimeMs, timing.streamLatencyMs);
      
      // Update progress indicator
      updateProgress(index, 1000);
      
      // Auto-scroll output container
      scrollOutputToBottom();
    },
    
    onMetricsUpdate: (metrics) => {
      // Update performance metrics display
      displayTokensPerSecond(metrics.tokensPerSecond);
      displayLatencyMs(metrics.averageLatencyMs);
      displayMemoryUsage(metrics.currentMemoryUsageMb);
    },
    
    onStatusChange: (status) => {
      // Handle various streaming states
      updateStatusUI(status); // "starting", "streaming", "paused", "completed", "error"
    }
  }
);

// Process tokens as they arrive with advanced control
try {
  let tokenCount = 0;
  for await (const token of stream) {
    // Update UI with each token
    document.getElementById("output").textContent += token.text;
    tokenCount++;
    
    // Check if user requested to pause
    if (isPauseRequested()) {
      await stream.pause();
      document.getElementById("status").textContent = "Paused";
      
      // Wait until resume is requested
      await waitForResumeRequest();
      await stream.resume();
      document.getElementById("status").textContent = "Resumed";
    }
    
    // Check if user requested to cancel
    if (isCancelRequested()) {
      await stream.cancel("User requested cancellation");
      document.getElementById("status").textContent = "Cancelled";
      break;
    }
    
    // Custom post-processing for specific tokens
    if (token.text.includes("quantum")) {
      highlightWord("quantum");
    }
  }
  
  document.getElementById("status").textContent = "Completed";
  document.getElementById("tokenCount").textContent = `${tokenCount} tokens generated`;
  
} catch (error) {
  console.error("Streaming error:", error);
  document.getElementById("status").textContent = `Error: ${error.message}`;
}

// Add regeneration feature
document.getElementById("regenerateButton").addEventListener("click", async () => {
  const currentOutput = document.getElementById("output").textContent;
  const lastNewlineIndex = currentOutput.lastIndexOf("\n");
  
  // Trim to last complete paragraph
  const trimmedPrompt = lastNewlineIndex > 0 
    ? currentOutput.substring(0, lastNewlineIndex) 
    : "";
  
  // Clear the rest and continue generation
  document.getElementById("output").textContent = trimmedPrompt;
  
  // Start new stream from this point
  const continuationStream = await model.generateStream(
    trimmedPrompt,
    { maxTokens: 500, temperature: 0.8 }
  );
  
  // Process continuation
  for await (const token of continuationStream) {
    document.getElementById("output").textContent += token;
  }
});
```

### Sophisticated Memory Management

```javascript
// Create memory management profile for large models in constrained environments
const memoryProfile = {
  // Core memory limits
  maxTotalMemoryMb: 3072,
  reserveSystemMemoryMb: 512,
  emergencyBufferMb: 256,
  
  // Component-specific memory policies
  componentPolicies: {
    "embeddings": { memoryPriority: "critical", minPrecisionBits: 8 },
    "first_layers": { memoryPriority: "high", minPrecisionBits: 4 },
    "middle_layers": { 
      memoryPriority: "medium", 
      minPrecisionBits: 2,
      swappingEnabled: true,
      swapThresholdMs: 10000  // Swap after 10 seconds of inactivity
    },
    "kv_cache": {
      memoryPriority: "adaptive",  // Priority changes based on usage
      pruningEnabled: true,
      pruningStrategy: "sliding_window",
      windowSize: 2048,
      compressionEnabled: true
    }
  },
  
  // Memory monitoring
  monitoring: {
    enabled: true,
    intervalMs: 500,
    thresholds: {
      warning: 0.7,  // 70% of max memory
      critical: 0.85, // 85% of max memory
      emergency: 0.95 // 95% of max memory
    },
    adaptationPolicy: "proactive" // Start adapting at warning threshold
  },
  
  // Memory pressure responses
  pressureActions: {
    warning: ["compress_kv_cache", "defer_tensor_allocations"],
    critical: ["reduce_precision", "unload_inactive_components", "prune_kv_cache"],
    emergency: ["force_component_unloading", "reduce_batch_size", "pause_generation"]
  },
  
  // Optimization strategies
  optimizations: {
    enableTensorReuse: true,
    enableLazyTensorAllocation: true,
    enableGarbageCollection: true,
    gcIntervalMs: 5000,
    tensorFragmentationLimit: 0.2,
    enableMemoryDefragmentation: true
  }
};

// Apply memory profile to model
await model.applyMemoryProfile(memoryProfile);

// Register memory status callbacks
model.onMemoryStatus((status) => {
  // Update UI with memory status
  updateMemoryUsageUI(status.currentUsageMb, status.maxMemoryMb, status.usagePercentage);
  
  // Show warnings when approaching limits
  if (status.pressureLevel === "critical") {
    showWarningNotification("Memory usage critical", 
      `Memory usage at ${status.usagePercentage.toFixed(1)}%. Performance may degrade.`);
  }
  
  // Log detailed component usage
  console.log("Component memory usage:", status.componentUsage);
});

// Get detailed memory analytics
const memoryAnalytics = await model.getMemoryAnalytics();
displayMemoryChart(memoryAnalytics);

// Memory usage over time for profiling
const memoryTimeSeries = await model.getMemoryTimeSeries({
  duration: "last_10_minutes",
  resolution: "1_second"
});
displayMemoryTimeSeriesChart(memoryTimeSeries);

// Manually trigger memory optimization when needed
document.getElementById("optimizeMemoryButton").addEventListener("click", async () => {
  const result = await model.optimizeMemory({
    target: "aggressive",
    allowComponentUnloading: true,
    allowPrecisionReduction: true
  });
  
  alert(`Memory optimized: Released ${result.freedMemoryMb}MB`);
});
```

### Advanced Browser Adaptation and Performance Tuning

```javascript
// Create comprehensive browser adaptation strategy
const adaptationStrategy = await accelerator.createAdaptationStrategy({
  // Detailed browser capabilities detection
  detectionMode: "comprehensive", // Basic, standard, or comprehensive
  detectExperimentalFeatures: true,
  detectHardwareAcceleration: true,
  detectPowerState: true,
  detectNetworkConditions: true,
  detectAvailableMemory: true,
  detectCPUCores: true,
  
  // Browser-specific optimizations
  browserOptimizations: {
    "chrome": {
      minVersion: 113,
      preferredRendererBackend: "webgpu",
      shaderPrecision: "highp",
      workgroupSize: [128, 1, 1],
      useSharedArrayBuffer: true,
      useOffscreenCanvas: true,
      useComputePressure: true
    },
    "firefox": {
      minVersion: 118,
      preferredRendererBackend: "webgpu",
      shaderPrecision: "highp",
      workgroupSize: [256, 1, 1], // Firefox performs better with larger workgroups
      audioModelOptimizations: true // Firefox-specific audio model optimizations
    },
    "safari": {
      minVersion: 17.0,
      preferredRendererBackend: "webgl2", // Fall back to WebGL2 for better compatibility
      useWasmFallbacks: true,
      metalOptimizations: true,
      shaderPrecision: "mediump",
      conservativeMemoryUsage: true
    },
    "edge": {
      // Inherit Chrome optimizations
      inheritFrom: "chrome"
    }
  },
  
  // Hardware-specific optimizations
  hardwareOptimizations: {
    "nvidia": {
      preferLargerWorkgroups: true,
      useComputeShaders: true,
      preferredTensorLayout: "nhwc"
    },
    "amd": {
      workgroupSizeFactor: 0.5, // Smaller workgroups for AMD
      preferredTensorLayout: "nchw"
    },
    "intel": {
      conservativeWorkgroups: true,
      aggressiveKernelFusion: false
    },
    "apple": {
      metalPerformanceShadersEnabled: true,
      preferSmallerWorkgroups: true
    },
    "mobile": {
      powerSavingMode: true,
      reducedPrecision: true,
      aggressiveMemoryManagement: true
    }
  },
  
  // Dynamic adaptation rules
  dynamicAdaptation: {
    enabled: true,
    adaptToMemoryPressure: true,
    adaptToBatteryStatus: true,
    adaptToThermalStatus: true,
    adaptToNetworkConditions: true,
    adaptToPerformanceFeedback: true,
    
    // Performance targets
    targetTokensPerSecond: 15,
    minAcceptableTokensPerSecond: 5,
    targetFirstTokenLatencyMs: 500,
    targetModelLoadTimeSeconds: 5,
    
    // Adaptation thresholds
    memoryPressureThreshold: 0.8,
    batteryLevelThreshold: 0.2,
    thermalPressureThreshold: 0.7,
    
    // Adaptation actions
    actions: [
      { condition: "memory_pressure > 0.8", action: "reduce_batch_size" },
      { condition: "battery_level < 0.2", action: "enable_power_saving" },
      { condition: "thermal_pressure > 0.7", action: "reduce_compute_intensity" },
      { condition: "tokens_per_second < 5", action: "reduce_model_complexity" },
      { condition: "network_type == 'slow-2g'", action: "enable_offline_mode" }
    ]
  },
  
  // Forced overrides (if needed)
  forceFeatures: {
    disableWebGPU: false,
    forceWasmBackend: false,
    forceHighPrecision: false,
    forceLowPrecision: false
  }
});

// Apply the adaptation strategy
await accelerator.applyAdaptationStrategy(adaptationStrategy);

// Get detailed adaptation report
const adaptationReport = await accelerator.getAdaptationReport();
console.log("Adaptation decisions:", adaptationReport.decisions);
console.log("Enabled optimizations:", adaptationReport.enabledOptimizations);
console.log("Disabled features:", adaptationReport.disabledFeatures);
console.log("Performance impact:", adaptationReport.estimatedPerformanceImpact);

// Monitor adaptation events
accelerator.onAdaptationEvent((event) => {
  console.log(`Adaptation event: ${event.type}`, event.details);
  if (event.type === "feature_disabled" && event.details.feature === "webgpu") {
    showNotification("WebGPU disabled", "Falling back to WebAssembly for better compatibility");
  }
});

// Get hardware acceleration status
const accelerationStatus = await accelerator.getHardwareAccelerationStatus();
displayAccelerationStatus(accelerationStatus);
```

## Browser Compatibility

### Feature Support Matrix (Updated March 3, 2025)

| Feature | Chrome | Firefox | Edge | Safari | Mobile Chrome | Mobile Safari |
|---------|--------|---------|------|--------|---------------|---------------|
| WebGPU Basic | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited |
| Compute Shaders | ✅ Full | ✅ Full+ | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Shader Precompilation | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| 4-bit Quantization | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| 2/3-bit Quantization | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Progressive Loading | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| WebAssembly Fallback | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| WASM SIMD | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Unified Framework | 🔄 67% | 🔄 67% | 🔄 67% | 🔄 67% | 🔄 67% | 🔄 67% |
| Streaming Inference | 🔄 92% | 🔄 92% | 🔄 92% | 🔄 85% | 🔄 92% | 🔄 85% |
| WebSocket Streaming | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Adaptive Batch Sizing | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |

### Recommended Optimizations by Browser

| Browser | Recommended Optimizations |
|---------|---------------------------|
| Chrome | Shader precompilation, 2-bit quantization, compute shaders, unified framework, streaming inference |
| Firefox | Compute shaders (40% faster for audio), 3-bit quantization, streaming inference for LLMs |
| Edge | Same as Chrome |
| Safari | WebAssembly fallback, 4-bit quantization, progressive loading, unified framework |
| Mobile Chrome | 4-bit quantization, progressive loading, adaptive batch sizing |
| Mobile Safari | WebAssembly fallback, 8-bit quantization, aggressive memory management, unified framework |

## Performance Optimization

### Memory Efficiency Optimization

| Technique | Memory Reduction | Performance Impact | Quality Impact |
|-----------|------------------|-------------------|----------------|
| 4-bit Quantization | 75% | Minimal | Minimal (1-2%) |
| 3-bit Quantization | 81.25% | Minimal | Small (2-4%) |
| 2-bit Quantization | 87.5% | 5-10% slower | Moderate (5-7%) |
| Mixed Precision | 84% | Minimal | Minimal (1-3%) |
| Progressive Loading | 25-40% (peak) | Faster startup | None |
| KV Cache Optimization | 45-60% (for context) | Minimal | None |
| Unified Framework | Variable | 10-20% faster | None |
| Streaming Inference | 15-25% (peak) | Better UX | None |
| Adaptive Batch Sizing | None | 10-30% faster | None |

### Startup Time Optimization

| Technique | Improvement | Browser Support |
|-----------|-------------|----------------|
| Shader Precompilation | 30-45% faster | Chrome, Edge |
| Progressive Loading | 25-40% faster | All browsers |
| Parallel Component Loading | 30-45% faster for multimodal | All browsers |
| Weight Compression | 15-25% faster loading | All browsers |

## API Reference

### IPFSAccelerate

```typescript
class IPFSAccelerate {
  constructor(config?: AccelerateConfig);
  
  async loadModel(
    modelName: string, 
    options?: ModelLoadOptions
  ): Promise<Model>;
  
  getCapabilities(): BrowserCapabilities;
  
  adaptToBrowser(settings: BrowserSettings): void;
  
  getBenchmarkResults(): BenchmarkResults;
}
```

### Model

```typescript
interface Model {
  // Basic methods
  async generate(prompt: string, options?: GenerationOptions): Promise<string>;
  async generateStream(prompt: string, options?: GenerationOptions): Promise<TokenStream>;
  async embed(text: string): Promise<Float32Array>;
  
  // Memory management
  configureMemory(options: MemoryOptions): void;
  getMemoryUsage(): MemoryStats;
  
  // Component management
  getLoadedComponents(): string[];
  unloadComponent(componentName: string): void;
  prioritizeComponent(componentName: string, priority: number): void;
}
```

### WebPlatformAccelerator (NEW)

```python
class WebPlatformAccelerator:
    def __init__(
        model_path: str,                # Path to the model
        model_type: str,                # Type of model (text, vision, audio, multimodal)
        config: Dict[str, Any] = None,  # Configuration dictionary
        auto_detect: bool = True        # Automatically detect optimal features
    )
    
    # Create endpoint for inference
    def create_endpoint() -> Callable
    
    # Get metrics and information
    def get_performance_metrics() -> Dict[str, Any]
    def get_feature_usage() -> Dict[str, bool]
    def get_config() -> Dict[str, Any]
    def get_components() -> Dict[str, Any]
    def get_browser_compatibility_matrix() -> Dict[str, Dict[str, bool]]
```

### WebGPUStreamingInference (NEW)

```python
class WebGPUStreamingInference:
    def __init__(
        model_path: str,                # Path to the model
        config: Dict[str, Any] = None   # Configuration dictionary
    )
    
    # Generate with callback for streaming
    def generate(
        prompt: str,                   # Input prompt
        max_tokens: int = 100,         # Maximum tokens to generate
        temperature: float = 0.7,      # Sampling temperature
        callback: Callable = None      # Callback for each token
    ) -> str
    
    # Generate asynchronously
    async def generate_async(
        prompt: str,                   # Input prompt
        max_tokens: int = 100,         # Maximum tokens to generate
        temperature: float = 0.7       # Sampling temperature
    ) -> str
    
    # Stream over WebSocket
    async def stream_websocket(
        websocket,                     # WebSocket connection
        prompt: str,                   # Input prompt
        max_tokens: int = 100,         # Maximum tokens to generate
        temperature: float = 0.7       # Sampling temperature
    )
    
    # Get performance statistics
    def get_performance_stats() -> Dict[str, Any]
```

### Helper Functions (NEW)

```python
# Create web endpoint with a single function call
def create_web_endpoint(
    model_path: str,               # Path to the model
    model_type: str,               # Type of model
    config: Dict[str, Any] = None  # Optional configuration
) -> Callable

# Get optimal configuration for a model
def get_optimal_config(
    model_path: str,               # Path to the model
    model_type: str                # Type of model
) -> Dict[str, Any]

# Get browser capabilities
def get_browser_capabilities() -> Dict[str, Any]

# Optimize configuration for streaming
def optimize_for_streaming(
    config: Dict[str, Any]         # Base configuration
) -> Dict[str, Any]

# Start WebSocket server for streaming
async def start_websocket_server(
    model_path: str,               # Path to the model
    host: str = "localhost",       # Host to bind the server to
    port: int = 8765               # Port to bind the server to
)
```

### BrowserCapabilityDetector

```typescript
class BrowserCapabilityDetector {
  getCapabilities(): BrowserCapabilities;
  getOptimizationProfile(): OptimizationProfile;
  getFeatureSupport(featureName: string): boolean;
}
```

## Examples

### Running a Text Generation Model

```javascript
import { IPFSAccelerate } from 'ipfs-accelerate-web';

async function runTextGeneration() {
  const accelerator = new IPFSAccelerate({
    enableWebGPU: true,
    enableProgressiveLoading: true
  });
  
  // Create progress UI
  const progressBar = document.getElementById('progress');
  const statusText = document.getElementById('status');
  
  // Load model with progress reporting
  const model = await accelerator.loadModel('llama-2-7b', {
    precision: {
      mode: 'mixed',
      bits: {
        embedding: 8,
        feed_forward: 2,
        attention: 3,
        lm_head: 4
      }
    },
    onProgress: (progress, component) => {
      progressBar.value = progress * 100;
      statusText.textContent = `Loading ${component}...`;
    },
    onComponentLoaded: (component) => {
      console.log(`Component loaded: ${component}`);
    }
  });
  
  // Start generating text when critical components are ready
  await model.waitForMinimumViableModel();
  statusText.textContent = 'Model ready for inference';
  
  // Generate text
  const output = await model.generate('Explain how WebGPU works:', {
    maxTokens: 200,
    temperature: 0.7,
    topP: 0.9
  });
  
  document.getElementById('output').textContent = output;
}

runTextGeneration().catch(console.error);
```

### Streaming Example

```javascript
import { IPFSAccelerate } from 'ipfs-accelerate-web';

async function runStreamingExample() {
  const accelerator = new IPFSAccelerate();
  
  // Load model with ultra-low precision
  const model = await accelerator.loadModel('llama-2-7b', {
    precision: {
      mode: 'ultra-low',
      bits: 2,
      adaptive: true
    }
  });
  
  // Set up streaming text generation
  const outputElement = document.getElementById('output');
  outputElement.textContent = '';
  
  // Start streaming generation
  const prompt = 'Write a short story about a robot learning to paint:';
  outputElement.textContent = prompt;
  
  const stream = await model.generateStream(prompt, {
    maxTokens: 500,
    temperature: 0.8
  });
  
  // Process tokens as they arrive
  for await (const token of stream) {
    outputElement.textContent += token;
  }
}

runStreamingExample().catch(console.error);
```

## Troubleshooting

### Common Issues

#### WebGPU Initialization Failed

**Symptoms:** Model fails to load with WebGPU errors.

**Solutions:**
1. Ensure browser supports WebGPU (Chrome 113+, Edge 113+, Firefox 118+)
2. Enable experimental WebGPU flags in browser settings
3. Update graphics drivers
4. Fall back to WebAssembly:
   ```javascript
   const accelerator = new IPFSAccelerate({
     enableWebGPU: false,
     forceWasm: true
   });
   ```

#### Out of Memory Errors

**Symptoms:** Browser crashes or shows out of memory errors.

**Solutions:**
1. Use lower precision:
   ```javascript
   const model = await accelerator.loadModel('llama-2-7b', {
     precision: {
       mode: 'ultra-low',
       bits: 2
     }
   });
   ```
2. Enable more aggressive memory management:
   ```javascript
   model.configureMemory({
     monitorIntervalMs: 500,
     threshold: 80,
     aggressiveUnloading: true,
     kvCacheStrategy: 'sliding_window',
     windowSize: 1024
   });
   ```
3. Use a smaller model or reduce context length

#### Safari-Specific Issues

**Symptoms:** Poor performance or crashes on Safari.

**Solutions:**
1. Use WebAssembly fallback mode
2. Enable Safari-specific optimizations:
   ```javascript
   accelerator.adaptToBrowser({
     browserName: 'safari',
     optimizationsEnabled: true
   });
   ```
3. Use 4-bit or 8-bit precision instead of ultra-low precision

#### Slow First Inference

**Symptoms:** First inference takes much longer than subsequent ones.

**Solutions:**
1. Enable shader precompilation:
   ```javascript
   const accelerator = new IPFSAccelerate({
     enableShaderPrecompilation: true
   });
   ```
2. Use progressive loading to start inference earlier:
   ```javascript
   const model = await accelerator.loadModel('llama-2-7b', {
     progressiveLoading: true,
     startInferenceEarly: true
   });
   ```

### Debugging Tools

#### Performance Profiling

```javascript
// Enable detailed performance monitoring
accelerator.enablePerformanceMonitoring({
  detailed: true,
  reportToConsole: true
});

// Generate performance report
const report = await accelerator.generatePerformanceReport();
console.log(report);
```

#### Feature Compatibility Check

```javascript
// Check detailed feature compatibility
const compatibility = accelerator.checkCompatibility({
  detailed: true,
  includeExperimental: true
});

console.log(compatibility);
```

#### Unified Framework Issues

**Symptoms:** Issues when using the unified framework

**Solutions:**
1. Enable verbose logging to identify problematic components:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Test components individually when unified framework fails:
   ```python
   # Instead of using the unified framework
   from fixed_web_platform.browser_capability_detector import BrowserCapabilityDetector
   from fixed_web_platform.webgpu_wasm_fallback import setup_wasm_fallback
   
   # Use components directly
   detector = BrowserCapabilityDetector()
   fallback = setup_wasm_fallback(model_path, model_type)
   ```

3. Use environment variables to disable problematic features:
   ```python
   import os
   # Disable specific features
   os.environ["WEBGPU_SHADER_PRECOMPILE"] = "0"
   os.environ["WEBGPU_COMPUTE_SHADERS"] = "0"
   ```

#### Streaming Inference Issues

**Symptoms:** Slow or interrupted streaming, out of memory errors during streaming

**Solutions:**
1. Reduce batch size and disable adaptive batching:
   ```python
   config = optimize_for_streaming({
       "adaptive_batch_size": False,
       "max_batch_size": 2 
   })
   ```

2. Enable ultra-low latency mode for interactive applications:
   ```python
   config = optimize_for_streaming({
       "ultra_low_latency": True,
       "stream_buffer_size": 1
   })
   ```

3. For WebSocket connection issues:
   ```python
   # Use direct callback instead of WebSocket
   def token_callback(token, is_last=False):
       print(token, end="", flush=True)
   
   result = streaming_handler.generate(prompt, callback=token_callback)
   ```

4. For asynchronous streaming issues:
   ```python
   # Use synchronous API instead
   result = streaming_handler.generate(prompt)
   ```

## Support and Resources

- [Full Documentation](https://docs.ipfsaccelerate.io/web)
- [API Reference](https://docs.ipfsaccelerate.io/web/api)
- [Examples Repository](https://github.com/ipfs-accelerate/web-examples)
- [Support Forum](https://community.ipfsaccelerate.io/web)
- [Issue Tracker](https://github.com/ipfs-accelerate/web/issues)
- [Performance Benchmarks](https://benchmarks.ipfsaccelerate.io/web)