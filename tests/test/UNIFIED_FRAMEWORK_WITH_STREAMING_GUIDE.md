# Unified Web Framework with Streaming Inference Guide
**Updated March 5, 2025**

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Getting Started](#getting-started)
4. [Streaming Inference Capabilities](#streaming-inference-capabilities)
5. [Precision Options and Memory Optimization](#precision-options-and-memory-optimization)
6. [Browser Compatibility](#browser-compatibility)
7. [Advanced Use Cases](#advanced-use-cases)
8. [Web Application Integration](#web-application-integration)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

## Introduction

The Unified Web Framework with Streaming Inference combines our WebGPU acceleration capabilities with an efficient streaming architecture to enable token-by-token generation for large language models directly in web browsers. This framework represents a significant advancement in browser-based machine learning capabilities, allowing developers to run advanced models with optimized memory usage, adaptable precision, and seamless cross-browser compatibility.

Key features of the August 2025 release include:

- **WebGPU Unified Framework**: A cohesive API that integrates all web platform acceleration components
- **Streaming Inference Pipeline**: Token-by-token generation with WebSocket support
- **Ultra-Low Precision Quantization**: Support for 2-bit, 3-bit, and 4-bit precision
- **Memory-Efficient KV Cache**: Optimized for handling long context windows
- **Browser Capability Detection**: Automatic adaptation to different browsers
- **Progressive Model Loading**: Efficient component-wise loading and initialization

This guide will walk you through implementing the unified framework with streaming inference in your web applications, covering everything from basic setup to advanced optimization techniques.

## Architecture Overview

The unified web framework consists of several integrated components:

```
┌─────────────────────────────────────────────────────┐
│             WebPlatformAccelerator                  │
├─────────────┬──────────────┬───────────────────────┤
│ WebGPU      │ WebNN        │ WebAssembly           │
│ Acceleration│ Acceleration │ Fallback              │
├─────────────┴──────────────┴───────────────────────┤
│                 Core Components                     │
├─────────┬────────────┬──────────────┬──────────────┤
│Progressive│ Shader    │ Browser      │ Hardware     │
│Loading    │ Registry  │ Detector     │ Adapters     │
├─────────┴────────────┴──────────────┼──────────────┤
│          Optimization Components     │              │
├─────────┬────────────┬──────────────┤  Inference   │
│Ultra-Low │ KV Cache   │ Compute      │  Pipeline    │
│Precision │ Optimizer  │ Shaders      │              │
├─────────┴────────────┴──────────────┼──────────────┤
│         Streaming Components         │              │
├─────────┬────────────┬──────────────┤              │
│WebSocket │ Batch      │ Adaptive     │              │
│Server    │ Processing │ Generation   │              │
└─────────┴────────────┴──────────────┴──────────────┘
```

### Key Components

1. **WebPlatformAccelerator**: The main entry point providing a unified API for all web acceleration components.

2. **Acceleration Backends**:
   - **WebGPU Acceleration**: Primary acceleration method using the GPU via WebGPU
   - **WebNN Acceleration**: Neural network acceleration via WebNN API
   - **WebAssembly Fallback**: CPU-based fallback for browsers without GPU access

3. **Core Components**:
   - **Progressive Loading**: Loads model components in priority order
   - **Shader Registry**: Manages WebGPU shaders and precompilation
   - **Browser Detector**: Identifies capabilities and sets optimizations
   - **Hardware Adapters**: Specific optimizations for different GPUs

4. **Optimization Components**:
   - **Ultra-Low Precision**: 2-bit, 3-bit, and 4-bit quantization
   - **KV Cache Optimizer**: Memory-efficient cache for attention
   - **Compute Shaders**: Specialized operations for different model types

5. **Streaming Components**:
   - **WebSocket Server**: Real-time communication for token streaming
   - **Batch Processing**: Adaptive batch sizes during generation
   - **Adaptive Generation**: Runtime adaptation based on device performance

## Getting Started

### Prerequisites

- Modern web browser with WebGPU support (Chrome 115+, Edge 115+, Firefox 118+)
- For streaming inference: WebSocket support
- For optimal performance: GPU with compute shader support

### Basic Setup

First, import the necessary components:

```javascript
import { 
  WebPlatformAccelerator,
  createWebEndpoint,
  getOptimalConfig 
} from '@ipfs-accelerate/web-platform';

import { 
  WebGPUStreamingInference,
  createStreamingEndpoint 
} from '@ipfs-accelerate/streaming';
```

#### Standard Usage (Non-Streaming)

```javascript
async function setupModel() {
  // Get optimal configuration based on browser capabilities
  const config = await getOptimalConfig('llama-7b', 'text');
  
  // Create accelerator with automatic detection
  const accelerator = new WebPlatformAccelerator({
    modelPath: 'https://example.com/models/llama-7b',
    modelType: 'text',
    config: config,
    autoDetect: true
  });
  
  // Create inference endpoint
  const endpoint = accelerator.createEndpoint();
  
  // Run inference
  const result = await endpoint({
    text: "Explain the concept of WebGPU acceleration"
  });
  
  // Display result
  document.getElementById('output').textContent = result;
}
```

#### Streaming Inference Setup

```javascript
async function setupStreamingModel() {
  // Create streaming handler
  const streaming = new WebGPUStreamingInference({
    modelPath: 'https://example.com/models/llama-7b',
    config: {
      quantization: "int4",
      optimizeKVCache: true,
      latencyOptimized: true,
      adaptiveBatchSize: true
    }
  });
  
  // Define a callback function for tokens
  function handleToken(token, isLast) {
    document.getElementById('output').textContent += token;
    if (isLast) {
      console.log("Generation complete");
    }
  }
  
  // Run streaming generation
  const prompt = "Explain how WebGPU streaming works in browsers";
  
  streaming.generate({
    prompt: prompt,
    maxTokens: 100,
    temperature: 0.7,
    callback: handleToken
  });
}
```

#### Using the Unified Framework with Streaming

```javascript
async function setupUnifiedStreamingModel() {
  // Create accelerator with streaming enabled
  const accelerator = new WebPlatformAccelerator({
    modelPath: 'https://example.com/models/llama-7b',
    modelType: 'text',
    config: {
      streamingInference: true,
      quantization: 4,
      kvCacheOptimization: true
    },
    autoDetect: true
  });
  
  // Create streaming endpoint
  const endpoint = accelerator.createEndpoint();
  
  // Define token callback
  function handleToken(token, isLast) {
    document.getElementById('output').textContent += token;
  }
  
  // Run streaming inference
  await endpoint({
    text: "Explain WebGPU streaming inference",
    maxTokens: 100,
    temperature: 0.7,
    callback: handleToken
  });
}
```

## Streaming Inference Capabilities

The streaming inference pipeline enables token-by-token generation, providing a responsive user experience similar to services like ChatGPT or Claude. This section details the architecture and capabilities of our streaming implementation.

### Core Streaming Features

1. **Token-by-Token Generation**: Generate and display tokens as they're created, rather than waiting for the entire response.

2. **WebSocket Integration**: Real-time streaming to web clients using standard WebSocket protocol.

3. **Adaptive Batch Sizing**: Dynamically adjusts batch size based on device performance.

4. **Low-Latency Optimization**: Minimizes the time between tokens for a smooth experience.

5. **Memory-Efficient KV Cache**: Optimized attention cache for long generations.

6. **Compute/Transfer Overlap**: Parallel computation and data transfer operations to reduce latency.

7. **Advanced Token Prediction**: Predictive prefetching based on text patterns and confidence scoring.

### Implementation Options

#### 1. Callback-Based Streaming

The simplest approach using a callback function for each token:

```javascript
// Create streaming handler
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: { 
    quantization: "int4",
    computeTransferOverlap: true,  // Enable compute/transfer overlap
    tokenPredictionEnabled: true,  // Enable advanced token prediction
    adaptiveBatchSize: true        // Dynamically adjust batch size
  }
});

// Define callback
function tokenCallback(token, isLast) {
  console.log(token);
  document.getElementById('output').textContent += token;
  
  if (isLast) {
    console.log("Generation complete");
  }
}

// Run generation
streaming.generate({
  prompt: "Explain streaming inference",
  maxTokens: 100,
  callback: tokenCallback
});
```

#### 2. WebSocket-Based Streaming

For distributed applications or when the model runs on a separate server:

```javascript
// Client-side code
const socket = new WebSocket('ws://your-server/streaming');

// Set up event handlers
socket.onopen = () => {
  socket.send(JSON.stringify({
    prompt: "Explain streaming inference",
    maxTokens: 100,
    temperature: 0.7,
    precision: "int4"
  }));
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'token') {
    document.getElementById('output').textContent += data.token;
  } else if (data.type === 'complete') {
    console.log("Generation complete", data);
  }
};
```

Server-side code (Node.js example with Express):

```javascript
import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import { WebGPUStreamingInference } from '@ipfs-accelerate/streaming';

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// Set up WebSocket server
wss.on('connection', (ws) => {
  ws.on('message', async (message) => {
    const request = JSON.parse(message);
    
    // Create streaming handler
    const streaming = new WebGPUStreamingInference({
      modelPath: 'models/llama-7b',
      config: { 
        quantization: request.precision || "int4",
        optimizeKVCache: true
      }
    });
    
    // Use the stream_websocket method
    await streaming.streamWebsocket(
      ws,
      request.prompt,
      request.maxTokens,
      request.temperature
    );
  });
});

server.listen(8765, () => {
  console.log('Streaming server running on port 8765');
});
```

#### 3. Async Generator-Based Streaming

For modern JavaScript applications using async iterators:

```javascript
// Use the streaming endpoint with async iterator
const streamingEndpoint = createStreamingEndpoint({
  modelPath: 'models/llama-7b',
  config: { quantization: "int4" }
});

async function generateStream() {
  const prompt = "Explain streaming inference";
  const stream = streamingEndpoint.generateStream({
    prompt: prompt,
    maxTokens: 100
  });
  
  // Use the async iterator
  const output = document.getElementById('output');
  for await (const token of stream) {
    output.textContent += token;
  }
  
  console.log("Generation complete");
}
```

### Measuring Streaming Performance

To evaluate streaming performance, track these key metrics:

1. **Time to First Token (TTFT)**: Time from request to first token
2. **Inter-Token Latency**: Average time between tokens
3. **Tokens Per Second**: Generation throughput
4. **Memory Usage**: Peak memory during streaming

Example measurement code:

```javascript
let startTime = Date.now();
let tokenCount = 0;
let firstTokenTime = null;
let lastTokenTime = null;
let interTokenTimes = [];

function measureTokenCallback(token, isLast) {
  const now = Date.now();
  tokenCount++;
  
  // Measure time to first token
  if (tokenCount === 1) {
    firstTokenTime = now;
    console.log(`Time to first token: ${(firstTokenTime - startTime)}ms`);
  }
  
  // Measure inter-token latency
  if (lastTokenTime !== null) {
    interTokenTimes.push(now - lastTokenTime);
  }
  lastTokenTime = now;
  
  // Display token
  document.getElementById('output').textContent += token;
  
  // Final metrics
  if (isLast) {
    const totalTime = now - startTime;
    const avgInterTokenLatency = interTokenTimes.reduce((a, b) => a + b, 0) / interTokenTimes.length;
    const tokensPerSecond = tokenCount / (totalTime / 1000);
    
    console.log(`Generation complete:
      - Tokens: ${tokenCount}
      - Total time: ${totalTime}ms
      - Avg inter-token latency: ${avgInterTokenLatency.toFixed(2)}ms
      - Tokens per second: ${tokensPerSecond.toFixed(2)}`);
  }
}
```

## Precision Options and Memory Optimization

The unified framework supports multiple precision formats, from standard 16-bit floating point to ultra-low 2-bit precision. This section covers precision options and memory optimization techniques.

### Precision Options Comparison

| Precision | Memory Reduction | Quality Impact | Use Cases |
|-----------|------------------|----------------|-----------|
| FP16 (16-bit) | Baseline | None | Highest quality needs |
| INT8 (8-bit) | 50% | Minimal | General use, vision models |
| INT4 (4-bit) | 75% | Slight | Text generation, embeddings |
| INT3 (3-bit) | 81.25% | Moderate | Memory-constrained devices |
| INT2 (2-bit) | 87.5% | Noticeable | Extreme memory constraints |

### Selecting the Right Precision

```javascript
// Select precision based on device capabilities and model
function selectOptimalPrecision(modelType, memoryConstraint) {
  // Start with 4-bit as a reasonable default
  let precision = "int4";
  
  if (memoryConstraint === "severe") {
    // For extremely limited memory (e.g., low-end devices)
    precision = "int2";
  } else if (memoryConstraint === "moderate") {
    // For moderately limited memory
    precision = "int3";
  } else if (modelType === "vision" || modelType === "audio") {
    // Vision and audio models often need higher precision
    precision = "int8";
  }
  
  return precision;
}

// Example usage
const deviceMemory = navigator.deviceMemory || 4; // GB of RAM, fallback to 4GB
const memoryConstraint = deviceMemory <= 2 ? "severe" : 
                         deviceMemory <= 4 ? "moderate" : "none";
                         
const precision = selectOptimalPrecision("text", memoryConstraint);
console.log(`Selected precision: ${precision}`);
```

### KV Cache Optimization

The KV (Key-Value) cache is critical for efficient transformer inference, especially for long context windows. Our ultra-low precision KV cache reduces memory usage while maintaining generation quality.

```javascript
// KV cache configuration example
const kvCacheConfig = {
  // 2-bit, 3-bit, or 4-bit precision
  bits: 3,
  
  // Number of values to group together for quantization
  groupSize: 64,
  
  // Maximum sequence length
  maxSeqLen: 8192,
  
  // Progressive precision (higher precision for recent tokens)
  progressivePrecision: true
};

// Creating optimized KV cache
const streamingConfig = {
  quantization: "int4",
  optimizeKVCache: true,
  kvCacheConfig: kvCacheConfig,
  adaptiveBatchSize: true
};

const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: streamingConfig
});
```

### Memory Usage Monitoring

Monitoring memory usage is essential for optimization:

```javascript
// Request memory measurement during generation
const memoryUsage = await streaming.measureMemoryUsage({
  prompt: "Test prompt for memory measurement",
  maxTokens: 50
});

console.log("Memory Usage Statistics:");
console.log(`- Peak WebGPU Buffer Memory: ${memoryUsage.peakWebGPUBufferMB} MB`);
console.log(`- KV Cache Size: ${memoryUsage.kvCacheMB} MB`);
console.log(`- Model Weight Memory: ${memoryUsage.modelWeightsMB} MB`);
console.log(`- Total Estimated Memory: ${memoryUsage.totalEstimatedMB} MB`);

// Memory efficiency with different precision
console.log(`- Memory reduction: ${memoryUsage.memoryReductionPercent}%`);
console.log(`- Original FP16 equivalent memory: ${memoryUsage.originalFP16MB} MB`);
```

### Adaptive Precision Configuration

For optimal memory efficiency with minimal quality loss, use adaptive precision:

```javascript
// Adaptive precision configuration
const adaptivePrecisionConfig = {
  // Enable mixed precision
  mixedPrecision: true,
  
  // Layer-specific precision assignments
  layerPrecisions: {
    "embedding": 8,    // Higher precision for embedding
    "attention": 3,    // Ultra-low precision for attention
    "feedforward": 4,  // Standard low precision for FF
    "lmhead": 8        // Higher precision for output
  },
  
  // Dynamic adaptation based on token importance
  dynamicAdaptation: true,
  
  // KV cache mixed precision (higher for recent tokens)
  kvCacheMixedPrecision: true
};

// Create streaming handler with adaptive precision
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: {
    quantization: "adaptive",
    adaptivePrecisionConfig: adaptivePrecisionConfig,
    optimizeKVCache: true
  }
});
```

## Browser Compatibility

The unified framework is designed to work across major browsers, with fallback mechanisms for partial or missing WebGPU support. This section covers browser compatibility and optimization techniques for different browsers.

### Browser Compatibility Matrix

| Feature | Chrome 115+ | Edge 115+ | Firefox 118+ | Safari 17.4+ | Mobile Chrome | Mobile Safari |
|---------|------------|-----------|-------------|-------------|---------------|--------------|
| WebGPU Support | ✅ Full | ✅ Full | ✅ Full | ⚠️ Partial | ✅ Full | ⚠️ Partial |
| WebNN Support | ✅ Full | ✅ Full | ❌ None | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial |
| Compute Shaders | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Shader Precompilation | ✅ Full | ✅ Full | ⚠️ Limited | ❌ None | ✅ Full | ❌ None |
| Ultra-Low Precision | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Compute/Transfer Overlap | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Advanced Token Prediction | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| WebAssembly SIMD | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| WebSockets | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| 2-bit/3-bit Precision | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Streaming Performance | ✅ High | ✅ High | ✅ High | ⚠️ Medium | ⚠️ Medium | ⚠️ Low |

### Browser Detection and Adaptation

The framework automatically detects browser capabilities and adapts accordingly:

```javascript
// The framework handles this automatically, but you can 
// access browser detection information:
const capabilities = await WebPlatformAccelerator.detectCapabilities();

console.log("Browser Capabilities:");
console.log(`- Browser: ${capabilities.browser} ${capabilities.version}`);
console.log(`- WebGPU Support: ${capabilities.webgpu.available ? "Yes" : "No"}`);
console.log(`- Compute Shaders: ${capabilities.webgpu.computeShaders ? "Yes" : "No"}`);
console.log(`- WebNN Support: ${capabilities.webnn.available ? "Yes" : "No"}`);
console.log(`- WebGL Fallback: ${capabilities.webgl.available ? "Yes" : "No"}`);
console.log(`- WebAssembly SIMD: ${capabilities.wasm.simd ? "Yes" : "No"}`);
```

### Browser-Specific Optimizations

#### Chrome/Edge

Chrome and Edge provide the best WebGPU support and performance:

```javascript
// Chrome/Edge optimized configuration
const chromeConfig = {
  shaderPrecompilation: true,
  computeShaders: true,
  workgroupSize: [128, 1, 1],  // Optimal for Chrome/Edge
  parallelCompilation: true,
  quantization: "int3"  // Chrome/Edge perform well with 3-bit
};
```

#### Firefox

Firefox has excellent WebGPU support with some unique optimizations:

```javascript
// Firefox optimized configuration
const firefoxConfig = {
  shaderPrecompilation: true, 
  computeShaders: true,
  workgroupSize: [256, 1, 1],  // Firefox prefers larger workgroups
  parallelCompilation: false,  // Can be unstable in Firefox
  quantization: "int3"
};

// Firefox performs especially well with audio models
if (modelType === "audio") {
  firefoxConfig.computeShaderOptimization = "audio";
}
```

#### Safari

Safari requires special handling with its Metal API integration:

```javascript
// Safari optimized configuration
const safariConfig = {
  shaderPrecompilation: false,  // Not fully supported
  computeShaders: false,        // Limited support
  fallbackToWebGL: true,        // Use WebGL when needed
  parallelLoading: true,        // Safari benefits from this
  quantization: "int8",         // Higher precision for stability
  metalOptimizations: true      // Safari-specific Metal optimizations
};
```

#### Mobile Browsers

Mobile devices require memory and performance optimizations:

```javascript
// Mobile optimization configuration
const mobileConfig = {
  quantization: "int2",         // Ultra-low precision for memory
  progressiveLoading: true,     // Critical for mobile
  optimizeKVCache: true,        // Essential for memory efficiency
  reducedModelSize: true,       // Use smaller model variants
  adaptiveBatchSize: true,      // Adjust based on device performance
  lowLatencyMode: false,        // Prioritize efficiency over latency
  memoryOptimizedMode: true     // Aggressive memory optimization
};
```

### Feature Detection and Fallbacks

The framework implements automatic fallbacks, but you can control this process:

```javascript
// Configure fallback behavior
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  config: {
    // Fallback preferences (in order)
    fallbackPreference: ["webgpu", "webnn", "webgl", "wasm"],
    
    // Specific fallback configurations
    fallbacks: {
      webgl: {
        precision: "int8",
        optimizations: ["textureBasedCompute", "packing"]
      },
      wasm: {
        threads: true,
        simd: true,
        memoryGrowthStrategy: "conservative"
      }
    }
  }
});
```

## Advanced Use Cases

This section covers advanced use cases for the unified framework and streaming inference pipeline.

### Long Context Generation

Handling long context windows effectively:

```javascript
// Configure for long context generation
const longContextConfig = {
  quantization: "int3",        // Lower precision for memory efficiency
  optimizeKVCache: true,
  kvCacheConfig: {
    bits: 3,                   // Ultra-low precision KV cache
    maxSeqLen: 32768,          // Very long context
    progressivePrecision: true // Higher precision for recent tokens
  },
  contextCompressionEnabled: true, // Enable attention compression
  contextCompressionRatio: 4,     // 4:1 compression for old tokens
  streamingChunkSize: 4096        // Process in manageable chunks
};

// Create streaming handler for long context
const longContextStreaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b-32k',
  config: longContextConfig
});

// Run with a long document
const document = await fetch('/api/long-document').then(r => r.text());
longContextStreaming.generate({
  prompt: `Summarize this document:\n\n${document}`,
  maxTokens: 500,
  callback: (token) => { /* handle token */ }
});
```

### Multimodal Streaming

Handling multimodal inputs with streaming text generation:

```javascript
// Multimodal (text + image) streaming
const multimodalConfig = {
  quantization: "int4",
  parallelLoading: true,      // Load components in parallel
  progressiveLoading: true,   // Critical for multimodal
  optimizeKVCache: true,
  computeTransferOverlap: true,  // Enable compute/transfer overlap
  tokenPredictionEnabled: true,  // Enable advanced token prediction
  visionModelPath: "models/clip-vit-base"  // Vision encoder
};

// Create multimodal streaming handler
const multimodalStreaming = new WebGPUStreamingInference({
  modelPath: 'models/llava-7b',
  config: multimodalConfig
});

// Process image
const imageData = await loadAndProcessImage("example.jpg");

// Run streaming generation with image
multimodalStreaming.generateWithImage({
  prompt: "Describe this image in detail",
  imageData: imageData,
  maxTokens: 300,
  callback: (token) => { /* handle token */ }
});
```

### Multi-Device Load Balancing

Distributing workload across multiple browser tabs or worker threads:

```javascript
// Coordinator code
class ModelShardingCoordinator {
  constructor(modelPath, numShards) {
    this.modelPath = modelPath;
    this.numShards = numShards;
    this.shards = [];
    this.isInitialized = false;
  }
  
  async initialize() {
    // Create sharded windows/workers
    for (let i = 0; i < this.numShards; i++) {
      const shardWindow = window.open('/model-worker.html', `shard-${i}`);
      this.shards.push(shardWindow);
      
      // Wait for window to load
      await new Promise(resolve => {
        window.addEventListener('message', function onMsg(e) {
          if (e.data.type === 'shard-ready' && e.data.shardId === i) {
            window.removeEventListener('message', onMsg);
            resolve();
          }
        });
      });
      
      // Initialize shard
      shardWindow.postMessage({
        type: 'initialize',
        shardId: i,
        totalShards: this.numShards,
        modelPath: this.modelPath
      }, '*');
    }
    
    this.isInitialized = true;
  }
  
  async generateStreaming(prompt, maxTokens, callback) {
    if (!this.isInitialized) await this.initialize();
    
    // Set up message handling for tokens
    window.addEventListener('message', (e) => {
      if (e.data.type === 'token') {
        callback(e.data.token, e.data.isLast);
      }
    });
    
    // Start generation on primary shard
    this.shards[0].postMessage({
      type: 'generate',
      prompt: prompt,
      maxTokens: maxTokens
    }, '*');
  }
}

// Worker page code (model-worker.html)
let shardId, totalShards, modelPath;
let streamingHandler;

window.addEventListener('message', async (e) => {
  if (e.data.type === 'initialize') {
    shardId = e.data.shardId;
    totalShards = e.data.totalShards;
    modelPath = e.data.modelPath;
    
    // Initialize this shard
    await initializeShard();
    
    // Signal ready
    window.opener.postMessage({
      type: 'shard-ready',
      shardId: shardId
    }, '*');
  }
  else if (e.data.type === 'generate') {
    // Run generation
    await runGeneration(e.data.prompt, e.data.maxTokens);
  }
});

async function initializeShard() {
  // Configure for this shard
  const config = {
    quantization: "int4",
    shardId: shardId,
    totalShards: totalShards
  };
  
  // Create streaming handler
  streamingHandler = new WebGPUStreamingInference({
    modelPath: modelPath,
    config: config
  });
}

async function runGeneration(prompt, maxTokens) {
  // Define callback to send tokens back to coordinator
  function handleToken(token, isLast) {
    window.opener.postMessage({
      type: 'token',
      token: token,
      isLast: isLast
    }, '*');
  }
  
  // Run streaming generation
  await streamingHandler.generate({
    prompt: prompt,
    maxTokens: maxTokens,
    callback: handleToken
  });
}
```

### Streaming with Real-Time Input

Building interactive applications with real-time streaming:

```javascript
// Interactive chat implementation
class InteractiveStreamingChat {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.streaming = null;
    this.conversation = [];
    this.isGenerating = false;
    this.stopRequested = false;
  }
  
  async initialize() {
    this.streaming = new WebGPUStreamingInference({
      modelPath: this.modelPath,
      config: {
        quantization: "int4",
        optimizeKVCache: true,
        adaptiveBatchSize: true
      }
    });
  }
  
  async sendMessage(userMessage) {
    if (this.isGenerating) {
      return { error: "Already generating a response" };
    }
    
    // Add user message to conversation
    this.conversation.push({ role: "user", content: userMessage });
    
    // Format conversation for the model
    const prompt = this.formatConversation();
    
    // Set up for generation
    this.isGenerating = true;
    this.stopRequested = false;
    let responseText = "";
    
    // Define token callback
    const handleToken = (token, isLast) => {
      responseText += token;
      
      // Call onToken callback if provided
      if (this.onToken) {
        this.onToken(token, isLast);
      }
      
      if (isLast || this.stopRequested) {
        this.isGenerating = false;
        
        // Add assistant response to conversation
        this.conversation.push({ role: "assistant", content: responseText });
        
        // Call onComplete callback if provided
        if (this.onComplete) {
          this.onComplete(responseText);
        }
      }
    };
    
    // Generate response
    try {
      await this.streaming.generate({
        prompt: prompt,
        maxTokens: 1000,
        temperature: 0.7,
        callback: handleToken
      });
      
      return { success: true };
    } catch (error) {
      this.isGenerating = false;
      return { error: error.message };
    }
  }
  
  stopGeneration() {
    if (this.isGenerating) {
      this.stopRequested = true;
      this.streaming.stopGeneration();
      return true;
    }
    return false;
  }
  
  formatConversation() {
    // Format conversation for the model
    let formattedPrompt = "";
    
    for (const message of this.conversation) {
      if (message.role === "user") {
        formattedPrompt += `User: ${message.content}\n\n`;
      } else if (message.role === "assistant") {
        formattedPrompt += `Assistant: ${message.content}\n\n`;
      }
    }
    
    formattedPrompt += "Assistant: ";
    return formattedPrompt;
  }
  
  // Set callbacks
  setCallbacks({ onToken, onComplete }) {
    this.onToken = onToken;
    this.onComplete = onComplete;
  }
}

// Usage example
const chat = new InteractiveStreamingChat('models/llama-7b-chat');
await chat.initialize();

// Set up callbacks
chat.setCallbacks({
  onToken: (token) => {
    document.getElementById('response').textContent += token;
  },
  onComplete: (response) => {
    console.log("Complete response:", response);
  }
});

// Send a message
document.getElementById('sendButton').addEventListener('click', async () => {
  const userMessage = document.getElementById('userInput').value;
  document.getElementById('response').textContent = "";
  
  const result = await chat.sendMessage(userMessage);
  if (result.error) {
    console.error(result.error);
  }
});

// Stop generation button
document.getElementById('stopButton').addEventListener('click', () => {
  chat.stopGeneration();
});
```

## Web Application Integration

Integrating the unified framework and streaming inference into web applications built with popular frameworks.

### React Integration

```jsx
// StreamingLLM.jsx
import React, { useState, useEffect, useRef } from 'react';
import { WebPlatformAccelerator } from '@ipfs-accelerate/web-platform';

const StreamingLLM = ({ modelPath, modelType }) => {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const outputRef = useRef(null);
  
  // Initialize model on component mount
  useEffect(() => {
    async function initializeModel() {
      try {
        const accelerator = new WebPlatformAccelerator({
          modelPath: modelPath,
          modelType: modelType,
          config: {
            streamingInference: true,
            quantization: "int4"
          },
          autoDetect: true
        });
        
        const endpoint = accelerator.createEndpoint();
        setModel({ accelerator, endpoint });
      } catch (err) {
        setError(`Error initializing model: ${err.message}`);
      }
    }
    
    initializeModel();
    
    // Cleanup on unmount
    return () => {
      // Release model resources if needed
    };
  }, [modelPath, modelType]);
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!model || isGenerating || !input.trim()) return;
    
    setIsGenerating(true);
    setOutput('');
    
    try {
      // Token callback function
      const handleToken = (token, isLast) => {
        setOutput(prev => prev + token);
        
        // Scroll to bottom as tokens arrive
        if (outputRef.current) {
          outputRef.current.scrollTop = outputRef.current.scrollHeight;
        }
        
        if (isLast) {
          setIsGenerating(false);
        }
      };
      
      // Run model with streaming
      await model.endpoint({
        text: input,
        maxTokens: 500,
        temperature: 0.7,
        callback: handleToken
      });
    } catch (err) {
      setError(`Generation error: ${err.message}`);
      setIsGenerating(false);
    }
  };
  
  // Stop generation
  const handleStop = () => {
    if (model && isGenerating) {
      model.accelerator.stopGeneration();
      setIsGenerating(false);
    }
  };
  
  return (
    <div className="streaming-llm">
      <form onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter your prompt here..."
          disabled={isGenerating}
        />
        <div className="buttons">
          <button type="submit" disabled={isGenerating || !model}>
            Generate
          </button>
          <button 
            type="button" 
            onClick={handleStop} 
            disabled={!isGenerating}
          >
            Stop
          </button>
        </div>
      </form>
      
      <div className="output" ref={outputRef}>
        {output || (isGenerating ? "Generating..." : "Output will appear here")}
      </div>
      
      {error && <div className="error">{error}</div>}
      
      {model && (
        <div className="status">
          Model loaded: {modelPath.split('/').pop()}
        </div>
      )}
    </div>
  );
};

export default StreamingLLM;
```

Usage in a React application:

```jsx
// App.jsx
import React from 'react';
import StreamingLLM from './components/StreamingLLM';

function App() {
  return (
    <div className="App">
      <header>
        <h1>Streaming LLM Demo</h1>
      </header>
      
      <main>
        <StreamingLLM 
          modelPath="https://example.com/models/llama-7b" 
          modelType="text" 
        />
      </main>
      
      <footer>
        <p>Powered by WebGPU Unified Framework</p>
      </footer>
    </div>
  );
}

export default App;
```

### Vue.js Integration

```vue
<!-- StreamingLLM.vue -->
<template>
  <div class="streaming-llm">
    <form @submit.prevent="handleSubmit">
      <textarea
        v-model="input"
        placeholder="Enter your prompt here..."
        :disabled="isGenerating"
      ></textarea>
      <div class="buttons">
        <button type="submit" :disabled="isGenerating || !modelReady">
          Generate
        </button>
        <button 
          type="button" 
          @click="handleStop" 
          :disabled="!isGenerating"
        >
          Stop
        </button>
      </div>
    </form>
    
    <div class="output" ref="outputContainer">
      {{ output || (isGenerating ? "Generating..." : "Output will appear here") }}
    </div>
    
    <div v-if="error" class="error">{{ error }}</div>
    
    <div v-if="modelReady" class="status">
      Model loaded: {{ modelPath.split('/').pop() }}
    </div>
  </div>
</template>

<script>
import { WebPlatformAccelerator } from '@ipfs-accelerate/web-platform';

export default {
  name: 'StreamingLLM',
  props: {
    modelPath: {
      type: String,
      required: true
    },
    modelType: {
      type: String,
      default: 'text'
    }
  },
  data() {
    return {
      input: '',
      output: '',
      isGenerating: false,
      accelerator: null,
      endpoint: null,
      modelReady: false,
      error: null
    };
  },
  mounted() {
    this.initializeModel();
  },
  beforeUnmount() {
    // Release model resources if needed
  },
  methods: {
    async initializeModel() {
      try {
        this.accelerator = new WebPlatformAccelerator({
          modelPath: this.modelPath,
          modelType: this.modelType,
          config: {
            streamingInference: true,
            quantization: "int4"
          },
          autoDetect: true
        });
        
        this.endpoint = this.accelerator.createEndpoint();
        this.modelReady = true;
      } catch (err) {
        this.error = `Error initializing model: ${err.message}`;
      }
    },
    async handleSubmit() {
      if (!this.modelReady || this.isGenerating || !this.input.trim()) return;
      
      this.isGenerating = true;
      this.output = '';
      
      try {
        await this.endpoint({
          text: this.input,
          maxTokens: 500,
          temperature: 0.7,
          callback: this.handleToken
        });
      } catch (err) {
        this.error = `Generation error: ${err.message}`;
        this.isGenerating = false;
      }
    },
    handleToken(token, isLast) {
      this.output += token;
      
      // Scroll output container to bottom
      this.$nextTick(() => {
        const container = this.$refs.outputContainer;
        if (container) {
          container.scrollTop = container.scrollHeight;
        }
      });
      
      if (isLast) {
        this.isGenerating = false;
      }
    },
    handleStop() {
      if (this.isGenerating) {
        this.accelerator.stopGeneration();
        this.isGenerating = false;
      }
    }
  }
};
</script>

<style scoped>
.streaming-llm {
  max-width: 800px;
  margin: 0 auto;
}

textarea {
  width: 100%;
  height: 120px;
  padding: 10px;
  margin-bottom: 10px;
  border-radius: 4px;
  border: 1px solid #ccc;
  font-family: inherit;
}

.buttons {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

button {
  padding: 8px 16px;
  border-radius: 4px;
  border: none;
  background-color: #4caf50;
  color: white;
  cursor: pointer;
}

button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.output {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 15px;
  min-height: 200px;
  max-height: 400px;
  overflow-y: auto;
  background-color: #f9f9f9;
  white-space: pre-wrap;
  margin-bottom: 15px;
}

.error {
  color: #d32f2f;
  margin: 10px 0;
}

.status {
  color: #2e7d32;
  font-size: 0.9em;
}
</style>
```

### Web Component Integration

For framework-agnostic integration, use Web Components:

```javascript
// streaming-llm-element.js
import { WebPlatformAccelerator } from '@ipfs-accelerate/web-platform';

class StreamingLLMElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    
    // Create initial DOM structure
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          font-family: system-ui, -apple-system, sans-serif;
        }
        
        .container {
          max-width: 800px;
          margin: 0 auto;
        }
        
        textarea {
          width: 100%;
          height: 120px;
          padding: 10px;
          margin-bottom: 10px;
          border-radius: 4px;
          border: 1px solid #ccc;
        }
        
        .buttons {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
        }
        
        button {
          padding: 8px 16px;
          border-radius: 4px;
          border: none;
          background-color: #4caf50;
          color: white;
          cursor: pointer;
        }
        
        button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        
        .output {
          border: 1px solid #ddd;
          border-radius: 4px;
          padding: 15px;
          min-height: 200px;
          max-height: 400px;
          overflow-y: auto;
          background-color: #f9f9f9;
          white-space: pre-wrap;
          margin-bottom: 15px;
        }
        
        .error {
          color: #d32f2f;
          margin: 10px 0;
        }
        
        .status {
          color: #2e7d32;
          font-size: 0.9em;
        }
      </style>
      
      <div class="container">
        <form>
          <textarea placeholder="Enter your prompt here..."></textarea>
          <div class="buttons">
            <button type="submit">Generate</button>
            <button type="button" class="stop-btn" disabled>Stop</button>
          </div>
        </form>
        
        <div class="output">Output will appear here</div>
        <div class="error" hidden></div>
        <div class="status" hidden></div>
      </div>
    `;
    
    // Get DOM elements
    this.textarea = this.shadowRoot.querySelector('textarea');
    this.form = this.shadowRoot.querySelector('form');
    this.generateBtn = this.shadowRoot.querySelector('button[type="submit"]');
    this.stopBtn = this.shadowRoot.querySelector('.stop-btn');
    this.outputElement = this.shadowRoot.querySelector('.output');
    this.errorElement = this.shadowRoot.querySelector('.error');
    this.statusElement = this.shadowRoot.querySelector('.status');
    
    // Bind methods
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleStop = this.handleStop.bind(this);
    this.handleToken = this.handleToken.bind(this);
    
    // Add event listeners
    this.form.addEventListener('submit', this.handleSubmit);
    this.stopBtn.addEventListener('click', this.handleStop);
    
    // Component state
    this.accelerator = null;
    this.endpoint = null;
    this.isGenerating = false;
    this.modelReady = false;
  }
  
  static get observedAttributes() {
    return ['model-path', 'model-type', 'max-tokens', 'temperature'];
  }
  
  connectedCallback() {
    // Initialize when component is connected
    this.initializeModel();
  }
  
  attributeChangedCallback(name, oldValue, newValue) {
    if (oldValue === newValue) return;
    
    if (name === 'model-path' || name === 'model-type') {
      this.initializeModel();
    }
  }
  
  disconnectedCallback() {
    // Release resources
    this.form.removeEventListener('submit', this.handleSubmit);
    this.stopBtn.removeEventListener('click', this.handleStop);
  }
  
  async initializeModel() {
    const modelPath = this.getAttribute('model-path');
    const modelType = this.getAttribute('model-type') || 'text';
    
    if (!modelPath) return;
    
    try {
      this.generateBtn.disabled = true;
      this.outputElement.textContent = "Loading model...";
      
      this.accelerator = new WebPlatformAccelerator({
        modelPath: modelPath,
        modelType: modelType,
        config: {
          streamingInference: true,
          quantization: "int4"
        },
        autoDetect: true
      });
      
      this.endpoint = this.accelerator.createEndpoint();
      this.modelReady = true;
      
      this.generateBtn.disabled = false;
      this.outputElement.textContent = "Model loaded. Ready to generate.";
      
      this.statusElement.textContent = `Model loaded: ${modelPath.split('/').pop()}`;
      this.statusElement.hidden = false;
      
      // Dispatch model ready event
      this.dispatchEvent(new CustomEvent('model-ready', {
        bubbles: true,
        composed: true
      }));
    } catch (err) {
      this.showError(`Failed to load model: ${err.message}`);
    }
  }
  
  async handleSubmit(event) {
    event.preventDefault();
    
    if (!this.modelReady || this.isGenerating) return;
    
    const input = this.textarea.value.trim();
    if (!input) return;
    
    this.isGenerating = true;
    this.outputElement.textContent = "";
    this.generateBtn.disabled = true;
    this.stopBtn.disabled = false;
    this.errorElement.hidden = true;
    
    try {
      const maxTokens = parseInt(this.getAttribute('max-tokens') || '500', 10);
      const temperature = parseFloat(this.getAttribute('temperature') || '0.7');
      
      await this.endpoint({
        text: input,
        maxTokens: maxTokens,
        temperature: temperature,
        callback: this.handleToken
      });
    } catch (err) {
      this.showError(`Generation error: ${err.message}`);
      this.finishGeneration();
    }
  }
  
  handleToken(token, isLast) {
    this.outputElement.textContent += token;
    this.outputElement.scrollTop = this.outputElement.scrollHeight;
    
    // Dispatch token event
    this.dispatchEvent(new CustomEvent('token', {
      bubbles: true,
      composed: true,
      detail: { token, isLast }
    }));
    
    if (isLast) {
      this.finishGeneration();
      
      // Dispatch complete event
      this.dispatchEvent(new CustomEvent('generation-complete', {
        bubbles: true,
        composed: true,
        detail: { text: this.outputElement.textContent }
      }));
    }
  }
  
  handleStop() {
    if (this.isGenerating && this.accelerator) {
      this.accelerator.stopGeneration();
      this.finishGeneration();
      
      // Dispatch stopped event
      this.dispatchEvent(new CustomEvent('generation-stopped', {
        bubbles: true,
        composed: true
      }));
    }
  }
  
  finishGeneration() {
    this.isGenerating = false;
    this.generateBtn.disabled = false;
    this.stopBtn.disabled = true;
  }
  
  showError(message) {
    this.errorElement.textContent = message;
    this.errorElement.hidden = false;
    
    // Dispatch error event
    this.dispatchEvent(new CustomEvent('error', {
      bubbles: true,
      composed: true,
      detail: { message }
    }));
  }
}

// Register the custom element
customElements.define('streaming-llm', StreamingLLMElement);
```

Usage in HTML:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Streaming LLM Demo</title>
  <script type="module" src="./streaming-llm-element.js"></script>
</head>
<body>
  <h1>Streaming LLM Web Component Demo</h1>
  
  <streaming-llm
    model-path="https://example.com/models/llama-7b"
    model-type="text"
    max-tokens="500"
    temperature="0.7"
  ></streaming-llm>
  
  <script>
    // Event listeners (optional)
    const llmElement = document.querySelector('streaming-llm');
    
    llmElement.addEventListener('model-ready', () => {
      console.log('Model is ready to use');
    });
    
    llmElement.addEventListener('token', (event) => {
      console.log('Token received:', event.detail.token);
    });
    
    llmElement.addEventListener('generation-complete', (event) => {
      console.log('Generation complete:', event.detail.text);
    });
    
    llmElement.addEventListener('error', (event) => {
      console.error('Error:', event.detail.message);
    });
  </script>
</body>
</html>
```

## Performance Tuning

Optimizing the unified framework and streaming inference for maximum performance.

### Performance Optimization Checklist

1. **Precision Selection**
   - Choose the right precision for your model and browser
   - Use mixed/adaptive precision for critical models

2. **Memory Optimization**
   - Enable KV cache optimization
   - Use progressive loading for large models
   - Implement component unloading when not needed

3. **Shader Optimization**
   - Enable shader precompilation
   - Use compute shaders for compatible browsers
   - Apply browser-specific workgroup sizes

4. **Streaming Optimization**
   - Configure adaptive batch sizes
   - Balance latency vs. throughput 
   - Optimize prefill phase

### Measuring Performance

Use the built-in benchmarking tools:

```javascript
// Run comprehensive benchmark
const benchmark = await WebPlatformAccelerator.runBenchmark({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  
  // Test different precision options
  precisions: ["int2", "int3", "int4", "int8"],
  
  // Test browser features
  features: {
    shaderPrecompilation: [true, false],
    computeShaders: [true, false],
    parallelLoading: [true, false]
  },
  
  // Run multiple prompt lengths
  promptLengths: [10, 100, 1000],
  
  // Generate different lengths
  generateLengths: [10, 50, 200],
  
  // Repeat tests for statistical significance
  iterations: 3
});

console.log("Benchmark Results:", benchmark);

// Detailed metrics
console.log("Detailed Metrics by Precision:");
for (const precision of benchmark.precisions) {
  console.log(`\n${precision} Precision:`);
  console.log(`- Time to First Token: ${benchmark.results[precision].timeToFirstToken}ms`);
  console.log(`- Tokens Per Second: ${benchmark.results[precision].tokensPerSecond}`);
  console.log(`- Memory Usage: ${benchmark.results[precision].memoryMB}MB`);
  console.log(`- Peak Memory: ${benchmark.results[precision].peakMemoryMB}MB`);
  console.log(`- Optimal Config: ${JSON.stringify(benchmark.results[precision].optimalConfig)}`);
}

// Generate performance recommendation
const recommendation = benchmark.getRecommendation();
console.log("\nRecommended Configuration:");
console.log(recommendation);
```

### Device-Specific Optimizations

```javascript
// Optimize based on device class
function getDeviceClass() {
  // Check for device memory
  const deviceMemory = navigator.deviceMemory || 4; // fallback to 4GB
  
  // Check for CPU cores
  const cpuCores = navigator.hardwareConcurrency || 4; // fallback to 4 cores
  
  // GPU detection (simplified example)
  let gpuTier = "unknown";
  if (window.gpu) {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const desc = await adapter.requestAdapterInfo();
      // Simple GPU classification
      if (desc.description.includes("NVIDIA")) {
        gpuTier = "high";
      } else if (desc.description.includes("AMD") || desc.description.includes("Intel")) {
        gpuTier = "medium";
      } else {
        gpuTier = "low";
      }
    }
  }
  
  // Classify device
  if (deviceMemory >= 8 && cpuCores >= 8 && gpuTier === "high") {
    return "high-end";
  } else if (deviceMemory >= 4 && cpuCores >= 4 && gpuTier !== "low") {
    return "mid-range";
  } else {
    return "low-end";
  }
}

// Create device-optimized configuration
function createOptimizedConfig(modelType, deviceClass) {
  const baseConfig = {
    streamingInference: true,
    optimizeKVCache: true,
    adaptiveBatchSize: true
  };
  
  // Add device-specific optimizations
  switch (deviceClass) {
    case "high-end":
      return {
        ...baseConfig,
        quantization: modelType === "multimodal" ? "int4" : "int3",
        parallelLoading: true,
        shaderPrecompilation: true,
        computeShaders: true,
        workgroupSize: [256, 1, 1]
      };
      
    case "mid-range":
      return {
        ...baseConfig,
        quantization: "int4",
        parallelLoading: modelType === "multimodal",
        shaderPrecompilation: true,
        computeShaders: true,
        workgroupSize: [128, 1, 1]
      };
      
    case "low-end":
      return {
        ...baseConfig,
        quantization: "int2",  // Ultra-low precision for memory
        memoryOptimizedMode: true,
        shaderPrecompilation: false, // Can be slow on low-end
        progressiveLoading: true,
        reducedModelSize: true
      };
      
    default:
      return baseConfig;
  }
}

// Usage
const deviceClass = await getDeviceClass();
console.log(`Detected device class: ${deviceClass}`);

const optimizedConfig = createOptimizedConfig("text", deviceClass);
console.log("Optimized configuration:", optimizedConfig);

// Create accelerator with optimized config
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  config: optimizedConfig,
  autoDetect: true  // Will still auto-detect, but with optimized defaults
});
```

### Advanced KV Cache Configuration

```javascript
// Optimizing KV cache for different use cases
function optimizeKVCache(contextLength, memoryConstraint, modelSize) {
  // Base configuration
  const baseConfig = {
    optimizeKVCache: true
  };
  
  // Determine appropriate bit precision
  let bits;
  if (memoryConstraint === "severe") {
    bits = 2;  // Ultra-low precision
  } else if (memoryConstraint === "moderate") {
    bits = 3;  // Good balance
  } else if (contextLength > 8192) {
    bits = 3;  // Good for very long context
  } else {
    bits = 4;  // Standard low precision
  }
  
  // Scale group size based on model size
  const groupSize = modelSize === "large" ? 128 :
                   modelSize === "medium" ? 64 : 32;
  
  // Progressive precision 
  const useProgressivePrecision = contextLength > 4096;
  
  // Enable attention compression for very long contexts
  const useAttentionCompression = contextLength > 16384;
  
  // Configure the KV cache
  const kvCacheConfig = {
    bits: bits,
    groupSize: groupSize,
    maxSeqLen: Math.max(contextLength * 1.5, 2048),  // Allow for growth
    progressivePrecision: useProgressivePrecision
  };
  
  // Add attention compression if needed
  if (useAttentionCompression) {
    kvCacheConfig.attentionCompression = {
      enabled: true,
      compressionRatio: contextLength > 32768 ? 8 : 4,
      compressTokensOlderThan: 4096
    };
  }
  
  return {
    ...baseConfig,
    kvCacheConfig: kvCacheConfig
  };
}

// Example usage
const longContextConfig = optimizeKVCache(32768, "moderate", "large");
console.log("Optimized KV cache for long context:", longContextConfig);

// Create streaming handler with optimized KV cache
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b-32k',
  config: {
    quantization: "int3",
    ...longContextConfig
  }
});
```

### Profiling and Optimization

Use the built-in profiling tools:

```javascript
// Enable performance profiling
const profilingConfig = {
  enableProfiling: true,
  profilingDetail: "high",
  collectLayerTiming: true,
  trackMemoryUsage: true,
  reportIntervalMs: 1000
};

// Create streaming with profiling
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: {
    quantization: "int4",
    ...profilingConfig
  }
});

// Run with profiling callback
streaming.generate({
  prompt: "Test prompt for profiling",
  maxTokens: 100,
  profilingCallback: (profilingData) => {
    console.log("Performance profile update:", profilingData);
    
    // Add to visualization (example)
    addToPerformanceChart(profilingData);
  }
});

// Get complete profile after generation
const profile = await streaming.getPerformanceProfile();

// Analyze profile to find bottlenecks
const bottlenecks = profile.findBottlenecks();
console.log("Performance bottlenecks:", bottlenecks);

// Get optimization suggestions
const suggestions = profile.generateOptimizationSuggestions();
console.log("Optimization suggestions:", suggestions);

// Apply suggested optimizations
const optimizedConfig = suggestions.apply({
  modelPath: 'models/llama-7b',
  modelType: 'text'
});

// Create new streaming handler with optimized config
const optimizedStreaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: optimizedConfig
});
```

## Troubleshooting

Common issues and solutions for the unified framework and streaming inference.

### Installation and Initialization Issues

| Issue | Solutions |
|-------|-----------|
| **Model fails to load** | 1. Check if the model path is correct<br>2. Ensure the browser has sufficient memory<br>3. Try a smaller model variant<br>4. Check browser console for specific errors |
| **WebGPU not available** | 1. Check if browser supports WebGPU<br>2. Update browser to latest version<br>3. Ensure hardware acceleration is enabled<br>4. Try using the WebAssembly fallback |
| **Out of memory errors** | 1. Use lower precision (2-bit or 3-bit)<br>2. Enable KV cache optimization<br>3. Use progressive loading<br>4. Reduce batch size<br>5. Try a smaller model |
| **Slow initialization** | 1. Enable shader precompilation<br>2. Use progressive loading<br>3. Check browser extensions that might interfere<br>4. Optimize model loading sequence |

### Streaming Issues

| Issue | Solutions |
|-------|-----------|
| **High latency between tokens** | 1. Enable latency optimization<br>2. Use appropriate batch size<br>3. Check for other CPU-intensive tasks<br>4. Ensure browser GPU priority is high |
| **Tokens stop unexpectedly** | 1. Check for maximum token limit reached<br>2. Ensure no memory errors occurred<br>3. Check if model supports the prompt length<br>4. Verify WebSocket connection stability |
| **WebSocket connection issues** | 1. Check server is running and accessible<br>2. Ensure CORS is properly configured<br>3. Verify WebSocket protocol compatibility<br>4. Check for network firewalls or proxies |
| **Browser freezing during generation** | 1. Reduce batch size<br>2. Enable progressive processing<br>3. Use Web Workers for background processing<br>4. Break generation into smaller chunks |

### Browser-Specific Issues

| Browser | Common Issues | Solutions |
|---------|---------------|-----------|
| **Chrome/Edge** | Performance degradation over time | 1. Enable GC-friendly memory management<br>2. Release unused resources explicitly<br>3. Use smaller workgroup sizes for long operations |
| **Firefox** | Shader compilation errors | 1. Use simpler compute shaders<br>2. Adjust workgroup size for Firefox<br>3. Disable parallel shader compilation |
| **Safari** | WebGPU feature limitations | 1. Use Metal-specific optimizations<br>2. Fall back to WebGL for unsupported features<br>3. Use higher precision for stability<br>4. Enable progressive fallbacks |
| **Mobile browsers** | Memory constraints | 1. Use 2-bit precision<br>2. Enable aggressive memory optimization<br>3. Use smaller model variants<br>4. Split work across multiple sessions |

### Quantization Issues

| Issue | Solutions |
|-------|-----------|
| **Quality degradation with low precision** | 1. Use 4-bit instead of 2-bit<br>2. Try adaptive precision<br>3. Increase precision for critical layers<br>4. Use mixed precision configuration |
| **Errors with ultra-low precision** | 1. Check browser compatibility with precision level<br>2. Verify model compatibility with precision level<br>3. Try different grouping strategies<br>4. Fall back to higher precision |
| **Slow performance with quantization** | 1. Check for optimized kernels for precision<br>2. Use compute shaders when possible<br>3. Ensure workgroup size is appropriate<br>4. Optimize quantization grouping strategy |

### Diagnostic Tools

```javascript
// Run diagnostic tests
const diagnostics = await WebPlatformAccelerator.runDiagnostics({
  features: ["webgpu", "webnn", "wasm", "streaming"],
  reportLevel: "detailed"
});

// Output diagnostics
console.log("Diagnostic Results:", diagnostics);

// Check for issues
if (diagnostics.issues.length > 0) {
  console.warn("Issues detected:", diagnostics.issues);
  
  // Apply automatic fixes where possible
  if (diagnostics.fixableIssues.length > 0) {
    const fixes = await WebPlatformAccelerator.applyAutomaticFixes(diagnostics.fixableIssues);
    console.log("Applied fixes:", fixes);
  }
  
  // Show recommendations for manual fixes
  for (const issue of diagnostics.manualFixIssues) {
    console.warn(`Issue: ${issue.description}`);
    console.warn(`Recommended action: ${issue.recommendedAction}`);
  }
}

// Output system information
console.log("Browser:", diagnostics.browserInfo);
console.log("WebGPU Support:", diagnostics.webgpuSupport);
console.log("WebNN Support:", diagnostics.webnnSupport);
console.log("WebAssembly Support:", diagnostics.wasmSupport);
console.log("Estimated Available Memory:", diagnostics.estimatedAvailableMemoryMB, "MB");
```

### Debugging Streaming Issues

```javascript
// Enable debug mode for streaming
const streamingDebug = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: {
    quantization: "int4",
    debug: true,      // Enable debug mode
    logLevel: "debug", // Set log level
    debugInference: true, // Debug inference steps
    debugTokens: true     // Debug token generation
  }
});

// Register debug callbacks
streamingDebug.setDebugCallbacks({
  onInferenceStep: (step) => {
    console.log(`Inference step: ${step.name}, time: ${step.timeMs}ms`);
  },
  onMemoryAllocation: (allocation) => {
    console.log(`Memory allocation: ${allocation.sizeBytes} bytes for ${allocation.purpose}`);
  },
  onTokenGeneration: (tokenInfo) => {
    console.log(`Token generated: "${tokenInfo.token}" (id: ${tokenInfo.tokenId}), probability: ${tokenInfo.probability}`);
  },
  onError: (error) => {
    console.error(`Error in streaming: ${error.message}`, error);
  }
});

// Debug generation
try {
  const result = await streamingDebug.generate({
    prompt: "Debug test prompt",
    maxTokens: 20,
    callback: (token) => {} // Regular callback
  });
  
  // Get debug information
  const debugInfo = streamingDebug.getDebugInfo();
  console.log("Debug Information:", debugInfo);
  
  // Export detailed log
  const debugLog = streamingDebug.exportDebugLog();
  console.log("Full Debug Log:", debugLog);
  
} catch (error) {
  console.error("Generation failed:", error);
}
```

## Conclusion

The Unified Web Framework with Streaming Inference represents a significant advancement in browser-based machine learning capabilities. By integrating WebGPU acceleration, ultra-low precision, and streaming token generation, this framework enables developers to build responsive, efficient AI-powered web applications.

Key aspects to remember:

1. Use the right precision for your model and use case
2. Leverage browser-specific optimizations
3. Implement proper memory management for large models
4. Take advantage of streaming for interactive applications
5. Use the diagnostic and profiling tools for optimization

For additional resources, check the following:

- [API Reference Documentation](https://docs.example.com/api)
- [Performance Optimization Guide](https://docs.example.com/performance)
- [Browser Compatibility Matrix](https://docs.example.com/compatibility)
- [Example Applications](https://github.com/example/web-platform-demos)
- [Troubleshooting Guide](https://docs.example.com/troubleshooting)

For help and community support:
- [Developer Forum](https://forum.example.com)
- [GitHub Issues](https://github.com/example/web-platform/issues)
- [Discord Community](https://discord.example.com)