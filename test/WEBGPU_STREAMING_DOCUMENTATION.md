# WebGPU Streaming Inference with Ultra-Low Precision

## Documentation & Developer Guide

**Version:** 0.9.0 (August 2025)

## Overview

WebGPU Streaming Inference enables token-by-token generation for large language models directly in web browsers with ultra-low precision quantization. This technology dramatically reduces memory usage while maintaining competitive performance, allowing even large models to run efficiently in browsers.

## Key Features

### Ultra-Low Precision Quantization
- **2-bit precision:** 87.5% memory reduction (8x longer context windows)
- **3-bit precision:** 81.25% memory reduction (5.3x longer context windows)
- **4-bit precision:** 75% memory reduction (4x longer context windows)
- **Quality-tuned quantization:** Layer-specific precision assignments for optimal quality

### Token-by-Token Streaming
- **Real-time token generation:** Immediate display of tokens as they're generated
- **Callback mechanism:** Flexible token handling with callback functions
- **WebSocket streaming:** Real-time streaming over WebSocket connections
- **Progress tracking:** Generation progress and metrics monitoring

### Memory-Efficient KV Cache
- **Ultra-low precision storage:** 2-bit, 3-bit, or 4-bit precision for KV cache
- **Context length extension:** 4-8x longer contexts with the same memory footprint
- **Progressive precision:** Higher precision for recent tokens, lower for older tokens
- **Efficient attention computation:** Optimized for WebGPU compute shaders

### Performance Optimization
- **Adaptive batch sizing:** Dynamically adjusts batch size based on device performance
- **Low-latency mode:** Optimized for interactive applications with minimal latency
- **Prefill optimization:** Faster initial response with optimized prefill phase
- **Compute/transfer overlap:** Parallel compute and transfer operations reduce latency
- **Advanced token prediction:** Predictive prefetching based on text patterns and prediction confidence
- **Browser-specific optimizations:** 
  - **Firefox audio optimizations:** 20-25% faster audio processing with 256x1x1 workgroup size
  - **Chrome text optimizations:** Enhanced text processing with 8x16 workgroups
  - **Edge vision optimizations:** Specialized vision processing with prefetch optimizations

### Cross-Platform Integration
- **Unified web framework:** Consistent API across all web platform components
- **Browser compatibility:** Works on all major browsers with WebGPU support
- **Feature detection:** Automatic detection and adaptation based on browser capabilities
- **Progressive enhancement:** Graceful fallbacks for different feature sets

## Getting Started

### Installation

```bash
# Using npm
npm install @ipfs-accelerate/web-platform

# Using yarn
yarn add @ipfs-accelerate/web-platform
```

### Basic Usage

#### Callback-Based Streaming

```javascript
import { WebGPUStreamingInference } from '@ipfs-accelerate/web-platform';

// Create streaming handler with 4-bit precision
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: {
    quantization: "int4",           // Use 4-bit precision
    optimizeKVCache: true,          // Enable KV cache optimization
    latencyOptimized: true,         // Optimize for low latency
    adaptiveBatchSize: true         // Dynamically adjust batch size
  }
});

// Define token callback
function handleToken(token, isLast) {
  document.getElementById('output').textContent += token;
  
  if (isLast) {
    console.log("Generation complete");
  }
}

// Generate text with streaming
streaming.generate({
  prompt: "Explain the concept of WebGPU streaming inference:",
  maxTokens: 100,
  temperature: 0.7,
  callback: handleToken
});
```

#### WebSocket Streaming

```javascript
// Client-side code
const socket = new WebSocket('ws://localhost:8765');

// Set up event handlers
socket.onopen = () => {
  socket.send(JSON.stringify({
    prompt: "Explain WebGPU streaming inference:",
    maxTokens: 100,
    temperature: 0.7,
    precision: "int4"  // Use 4-bit precision
  }));
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'token') {
    document.getElementById('output').textContent += data.token;
  }
  else if (data.type === 'complete') {
    console.log("Generation complete:", data);
  }
};
```

#### Using the Unified Framework

```javascript
import { WebPlatformAccelerator } from '@ipfs-accelerate/web-platform';

// Create accelerator with streaming configuration
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  config: {
    streamingInference: true,    // Enable streaming
    quantization: 4,             // Use 4-bit quantization
    kvCacheOptimization: true,   // Enable KV cache optimization
    latencyOptimized: true       // Optimize for low latency
  },
  autoDetect: true               // Auto-detect browser capabilities
});

// Create endpoint
const endpoint = accelerator.createEndpoint();

// Define token callback
function handleToken(token, isLast) {
  document.getElementById('output').textContent += token;
}

// Generate text with streaming
await endpoint({
  text: "Explain WebGPU streaming inference:",
  maxTokens: 100,
  temperature: 0.7,
  callback: handleToken
});
```

### React Component Integration

```jsx
import React, { useState } from 'react';
import { WebGPUStreamingExample } from '@ipfs-accelerate/web-platform-react';

function MyApp() {
  const [output, setOutput] = useState('');
  
  const handleToken = (token) => {
    setOutput(prev => prev + token);
  };
  
  return (
    <div className="app">
      <h1>WebGPU Streaming Demo</h1>
      
      <WebGPUStreamingExample 
        modelId="llama-7b"
        initialPrecision="4-bit"
        onToken={handleToken}
      />
      
      <div className="output">
        {output || "Output will appear here..."}
      </div>
    </div>
  );
}
```

## Advanced Configuration

### Ultra-Low Precision Options

```javascript
// Configure ultra-low precision
const ultraLowConfig = {
  quantization: "int2",              // 2-bit precision
  optimizeKVCache: true,
  kvCacheConfig: {
    bits: 2,                         // 2-bit KV cache
    groupSize: 64,                   // Group size for quantization
    maxSeqLen: 32768,                // Support for very long context
    progressivePrecision: true       // Higher precision for recent tokens
  },
  adaptivePrecision: true,           // Use mixed precision based on layer
  layerSpecificPrecision: {
    embedding: 4,                    // 4-bit for embeddings
    attention: 2,                    // 2-bit for attention
    feedforward: 3,                  // 3-bit for feedforward
    lmhead: 4                        // 4-bit for LM head
  }
};

// Create streaming handler with ultra-low precision
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: ultraLowConfig
});
```

### Configuration Validation and Auto-Correction

The framework includes a robust configuration validation and auto-correction system that ensures your settings are compatible with the current browser environment:

```javascript
import { WebPlatformAccelerator, ConfigurationManager } from '@ipfs-accelerate/web-platform';

// Create accelerator with auto-validation
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  config: {
    quantization: "2bit",           // This will be auto-corrected for Safari
    streamingInference: true,
    browser: "safari"               // Safari doesn't support 2-bit quantization
  },
  autoDetect: true                  // Enable auto-detection and validation
});

// Configuration is automatically corrected (4-bit for Safari)
const validatedConfig = accelerator.getConfig();
console.log("Validated config:", validatedConfig);

// Get detailed validation information using the ConfigurationManager
const configManager = new ConfigurationManager({
  modelType: 'text',
  browser: 'safari',
  autoCorrect: true
});

// Validate a configuration
const validationResult = configManager.validateConfiguration({
  quantization: "2bit",
  workgroupSize: "invalid"
});

console.log("Validation result:", validationResult);
// Shows validation errors and auto-corrected configuration

// For more information, see the Configuration Validation Guide
```

For comprehensive details, see the dedicated [Configuration Validation Guide](CONFIGURATION_VALIDATION_GUIDE.md).

### Performance Optimization

```javascript
// Configure for maximum performance
const performanceConfig = {
  quantization: "int4",
  optimizeKVCache: true,
  latencyOptimized: true,
  adaptiveBatchSize: true,
  maxBatchSize: 8,                   // Maximum batch size
  prefillOptimized: true,            // Optimize prefill phase
  streamBufferSize: 3,               // Token buffer size
  computeShaders: true,              // Use compute shaders
  workgroupSize: [256, 1, 1],        // Workgroup size for compute shaders
  parallelPrefill: true,             // Parallelize prefill phase
  cacheWarming: true,                // Prewarm WebGPU cache
  memoryOptimizedMode: true          // Aggressive memory optimization
};

// Create streaming handler with performance optimization
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: performanceConfig
});
```

### Browser-Specific Optimization

```javascript
import { detectBrowser, getOptimalConfig } from '@ipfs-accelerate/web-platform';

// Detect browser and get optimal configuration
const browserInfo = detectBrowser();
const optimalConfig = getOptimalConfig(
  'models/llama-7b',
  'text',
  browserInfo
);

console.log(`Using ${browserInfo.name} ${browserInfo.version}`);
console.log(`Optimal configuration:`, optimalConfig);

// Create accelerator with browser-specific optimization
const accelerator = new WebPlatformAccelerator({
  modelPath: 'models/llama-7b',
  modelType: 'text',
  config: optimalConfig,
  autoDetect: false  // Skip auto-detection, use provided config
});
```

#### Firefox Audio Optimization Example

Firefox provides superior performance for audio models with specialized 256x1x1 workgroup size configurations:

```javascript
import { optimizeForFirefox } from '@ipfs-accelerate/web-platform/audio';

// Create audio model with Firefox optimizations
const whisperConfig = {
  modelPath: 'models/whisper-small',
  modelType: 'audio',
  config: {
    firefoxOptimized: true,          // Enable Firefox-specific optimizations
    workgroupSize: [256, 1, 1],      // Use optimized workgroup size
    computeShaders: true,            // Enable compute shaders
    powerOptimized: true,            // Enable power optimization (15% less power)
    shaderPrecompilation: true       // Enable shader precompilation
  }
};

// Initialize model with Firefox optimizations
const whisperModel = optimizeForFirefox(whisperConfig);

// Extract features with 20-25% better performance than Chrome
const audioFeatures = await whisperModel.extractFeatures('audio.mp3');

// Get performance metrics
const metrics = whisperModel.getPerformanceMetrics();
console.log(`Firefox performance advantage: ${metrics.firefoxAdvantageOverChrome}`);
console.log(`Power impact: ${metrics.powerImpactPercent}%`);

## Performance Tracking

### Basic Metrics

```javascript
// Create streaming handler
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: { quantization: "int4" }
});

// Generate text
await streaming.generate({
  prompt: "Test prompt",
  maxTokens: 50,
  callback: (token) => console.log(token)
});

// Get performance metrics
const metrics = streaming.getPerformanceStats();
console.log("Performance Metrics:");
console.log(`- Time to First Token: ${metrics.timeToFirstToken}ms`);
console.log(`- Tokens Per Second: ${metrics.tokensPerSecond}`);
console.log(`- Generation Time: ${metrics.generationTime}s`);
console.log(`- Memory Usage: ${metrics.memoryUsage}MB`);
console.log(`- Batch Size History:`, metrics.batchSizeHistory);
```

### Detailed Profiling

```javascript
// Configure profiling
const profilingConfig = {
  quantization: "int4",
  profilingEnabled: true,             // Enable profiling
  profilingDetailLevel: "verbose",    // Full profiling detail
  memoryTracking: true,               // Track memory usage
  profilingCategories: [              // Categories to profile
    "tokenization",
    "inference",
    "kvcache",
    "memory"
  ]
};

// Create streaming handler with profiling
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b',
  config: profilingConfig
});

// Generate text with profiling callback
await streaming.generate({
  prompt: "Test prompt for profiling",
  maxTokens: 50,
  profilingCallback: (profilingData) => {
    console.log("Profiling update:", profilingData);
  }
});

// Get detailed profile
const profile = streaming.getDetailedProfile();
console.log("Detailed Profile:", profile);

// Generate optimization recommendations
const recommendations = streaming.getOptimizationRecommendations();
console.log("Optimization Recommendations:", recommendations);
```

## WebSocket Server Implementation

### Node.js Server

```javascript
import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import { WebGPUStreamingInference } from '@ipfs-accelerate/web-platform-node';

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// Set up WebSocket server
wss.on('connection', (ws) => {
  ws.on('message', async (message) => {
    const request = JSON.parse(message);
    
    // Create streaming handler
    const streaming = new WebGPUStreamingInference({
      modelPath: request.modelPath || 'models/llama-7b',
      config: { 
        quantization: request.precision || "int4",
        optimizeKVCache: true,
        latencyOptimized: true
      }
    });
    
    // Stream tokens over WebSocket
    await streaming.streamWebsocket(
      ws,
      request.prompt,
      request.maxTokens,
      request.temperature
    );
  });
});

// Start server
const PORT = process.env.PORT || 8765;
server.listen(PORT, () => {
  console.log(`Streaming server running on port ${PORT}`);
});
```

## Memory Efficiency

Ultra-low precision quantization significantly reduces memory usage, enabling larger models and longer context windows in browsers:

| Precision | Memory Reduction | 7B Model Size | Max Context Length |
|-----------|------------------|---------------|-------------------|
| FP16 (16-bit) | 0% (baseline) | ~14 GB | 4K tokens |
| INT8 (8-bit) | 50% | ~7 GB | 8K tokens |
| INT4 (4-bit) | 75% | ~3.5 GB | 16K tokens |
| INT3 (3-bit) | 81.25% | ~2.6 GB | 21K tokens |
| INT2 (2-bit) | 87.5% | ~1.75 GB | 32K tokens |

These memory reductions are critical for running large language models in web browsers where memory constraints are significant.

## Browser Compatibility

| Browser | WebGPU Support | Streaming | Ultra-Low Precision | Status | Specialized Optimizations |
|---------|----------------|-----------|---------------------|--------|---------------------------|
| Chrome 115+ | ✅ Full | ✅ Full | ✅ Full | Fully Supported | Enhanced text processing (8x16 workgroups) |
| Edge 115+ | ✅ Full | ✅ Full | ✅ Full | Fully Supported | Vision models with prefetch optimization |
| Firefox 118+ | ✅ Full | ✅ Full | ✅ Full | Fully Supported | **Audio models (20-25% faster, 256x1x1 workgroups)** |
| Safari 17.4+ | ⚠️ Partial | ⚠️ Limited | ⚠️ Limited | Limited Support | Conservative shader implementation (4x4 workgroups) |
| Mobile Chrome | ✅ Full | ⚠️ Limited | ✅ Full | Good Support | Power-optimized compute for mobile |
| Mobile Safari | ⚠️ Partial | ⚠️ Limited | ⚠️ Limited | Limited Support | Metal fallbacks |

For browsers with limited support, the system automatically falls back to more compatible options while maintaining functionality.

### Model-Browser Recommendations

Based on extensive benchmarking, we recommend these browser-model pairings for optimal performance:

| Model Type | Examples | Recommended Browser | Performance Advantage |
|------------|----------|---------------------|----------------------|
| Audio Models | Whisper, Wav2Vec2, CLAP | **Firefox** | 20-25% faster compute shaders, 15% less power usage |
| Text Models | BERT, T5, LLAMA, Qwen2 | Chrome/Edge | 10-15% faster with 8x16 workgroup size |
| Vision Models | ViT, DETR | Chrome/Edge | 15-20% faster with vision-optimized kernels |
| Multimodal Models | CLIP, LLaVA, XCLIP | Chrome/Edge | 10-15% faster parallel processing |

## Error Handling

```javascript
try {
  const streaming = new WebGPUStreamingInference({
    modelPath: 'models/llama-7b',
    config: { quantization: "int4" }
  });
  
  await streaming.generate({
    prompt: "Test prompt",
    maxTokens: 50,
    callback: (token) => console.log(token)
  });
} catch (error) {
  if (error.code === 'WEBGPU_NOT_SUPPORTED') {
    console.error("WebGPU is not supported in this browser");
    // Fall back to WebAssembly implementation
  } 
  else if (error.code === 'MODEL_LOADING_FAILED') {
    console.error("Failed to load model:", error.message);
    // Try loading a smaller model or with higher precision
  }
  else if (error.code === 'OUT_OF_MEMORY') {
    console.error("Out of memory error:", error.message);
    // Try with lower precision or smaller model
  }
  else {
    console.error("Unknown error:", error);
  }
}
```

## Advanced Use Cases

### Long Context Generation

```javascript
// Configure for long context generation
const longContextConfig = {
  quantization: "int3",              // Use 3-bit precision
  optimizeKVCache: true,
  kvCacheConfig: {
    bits: 3,                         // 3-bit KV cache
    maxSeqLen: 32768,                // Very long context
    progressivePrecision: true,      // Higher precision for recent tokens
    compressionEnabled: true,        // Enable KV cache compression
    pruningEnabled: true,            // Enable KV cache pruning
    compressionRatio: 4,             // 4:1 compression for old tokens
    compressionThreshold: 4096       // Compress tokens beyond this threshold
  }
};

// Create streaming handler for long context
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b-32k',
  config: longContextConfig
});

// Process a very long document with streaming
const longDocument = await fetch('/api/long-document').then(r => r.text());
let responseBuffer = '';

await streaming.generate({
  prompt: `Summarize this document:\n\n${longDocument}`,
  maxTokens: 500,
  callback: (token) => {
    responseBuffer += token;
    console.log(token);
  }
});
```

### Interactive Chat Application

```javascript
// Create streaming handler with optimized KV cache
const streaming = new WebGPUStreamingInference({
  modelPath: 'models/llama-7b-chat',
  config: {
    quantization: "int4",
    optimizeKVCache: true,
    latencyOptimized: true
  }
});

// Maintain conversation state
const conversation = [];

// Function to send a message and get streaming response
async function sendMessage(userMessage) {
  // Add user message to conversation
  conversation.push({ role: "user", content: userMessage });
  
  // Format conversation for the model
  let prompt = "";
  for (const message of conversation) {
    if (message.role === "user") {
      prompt += `User: ${message.content}\n\n`;
    } else {
      prompt += `Assistant: ${message.content}\n\n`;
    }
  }
  prompt += "Assistant: ";
  
  // Clear assistant response area
  document.getElementById('assistant-response').textContent = '';
  
  // Show typing indicator
  document.getElementById('typing-indicator').style.display = 'inline-block';
  
  // Generate response with streaming
  let assistantResponse = '';
  
  await streaming.generate({
    prompt: prompt,
    maxTokens: 500,
    temperature: 0.7,
    callback: (token, isLast) => {
      // Add token to response
      assistantResponse += token;
      
      // Update UI
      document.getElementById('assistant-response').textContent = assistantResponse;
      
      // Scroll to bottom
      const chatContainer = document.getElementById('chat-container');
      chatContainer.scrollTop = chatContainer.scrollHeight;
      
      // Hide typing indicator when complete
      if (isLast) {
        document.getElementById('typing-indicator').style.display = 'none';
      }
    }
  });
  
  // Add assistant response to conversation
  conversation.push({ role: "assistant", content: assistantResponse });
  
  return assistantResponse;
}

// Connect to UI
document.getElementById('send-button').addEventListener('click', () => {
  const userInput = document.getElementById('user-input').value;
  if (!userInput.trim()) return;
  
  // Display user message
  const userMessageElement = document.createElement('div');
  userMessageElement.className = 'user-message';
  userMessageElement.textContent = userInput;
  document.getElementById('chat-container').appendChild(userMessageElement);
  
  // Clear input
  document.getElementById('user-input').value = '';
  
  // Get assistant response
  sendMessage(userInput);
});
```

## Implementation Status

For current implementation status and upcoming features, see [Implementation Status](implementation_status.md).

## Additional Resources

### Developer Documentation

- [WebGPU Implementation Guide](docs/WEBGPU_IMPLEMENTATION_GUIDE.md) - Comprehensive guide to WebGPU integration
- [Developer Tutorial](docs/DEVELOPER_TUTORIAL.md) - Step-by-step tutorial with working examples
- [WebGPU Shader Precompilation Guide](docs/WEBGPU_SHADER_PRECOMPILATION.md) - Guide to shader precompilation optimization
- [Browser-Specific Optimizations](docs/browser_specific_optimizations.md) - Tailored configurations for different browsers
- [Firefox Audio Optimizations Guide](docs/FIREFOX_AUDIO_OPTIMIZATIONS.md) - Guide to Firefox's 20-25% faster audio processing
- [Error Handling Guide](docs/ERROR_HANDLING_GUIDE.md) - Comprehensive error handling strategy
- [Model-Specific Optimization Guides](docs/model_specific_optimizations/) - Guides for different model types
- [Audio Model Optimization Guide](docs/model_specific_optimizations/AUDIO_MODEL_GUIDE.md) - Specialized guide for audio model optimization

### Framework Resources

- [Unified Framework Guide](UNIFIED_FRAMEWORK_WITH_STREAMING_GUIDE.md) - Guide to the unified web framework
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Development roadmap and implementation status
- [WebGPU Streaming Demo](WebGPUStreamingDemo.html) - Interactive demo of streaming capabilities
- [Tutorial: Streaming Integration](tutorial_stream_integration.py) - Tutorial on integrating streaming
- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Configuration Validation Guide](CONFIGURATION_VALIDATION_GUIDE.md) - Guide to configuration validation