# IPFS Accelerate JavaScript SDK Documentation

**Date:** March 7, 2025  
**Version:** 0.4.0 (Current Release)  
**Target Complete Implementation:** October 15, 2025

## Overview

The IPFS Accelerate JavaScript SDK is a comprehensive toolkit for accelerating AI models in web browsers and Node.js environments. It provides a unified interface for leveraging WebNN, WebGPU, and WebAssembly hardware acceleration, optimizing content delivery through P2P networking, and storing benchmark results.

### Repository Structure

```
ipfs_accelerate_js/
├── src/
│   ├── worker/                # Core model execution components
│   │   ├── webnn/             # WebNN backend implementation
│   │   ├── webgpu/            # WebGPU backend implementation
│   │   ├── wasm/              # WebAssembly backend implementation
│   │   └── worker.js          # Worker management and hardware detection
│   ├── api_backends/          # API client implementations
│   │   ├── apis.js            # API client registry and factory
│   │   └── api_models_registry.js # Supported model mappings
│   ├── hardware/              # Hardware abstraction layer
│   │   ├── hardware_profile.js # Hardware profile definitions
│   │   ├── hardware_selector.js # Intelligent hardware selection
│   │   ├── backends/          # Hardware backend implementations
│   │   │   ├── webgpu_backend.js
│   │   │   ├── webnn_backend.js
│   │   │   └── wasm_backend.js
│   │   └── resource_pool.js   # Resource management
│   ├── utils/                 # Common utilities
│   │   ├── ipfs_multiformats.js # IPFS data handling utilities
│   │   └── config.js          # Configuration management
│   ├── model/                 # Model management
│   │   ├── model_accelerator.js
│   │   ├── model_registry.js
│   │   └── model_loader.js
│   ├── optimization/          # Optimization utilities
│   │   ├── optimization_engine.js
│   │   └── techniques/        # Optimization implementations
│   ├── quantization/          # Quantization framework
│   │   ├── quantization_engine.js
│   │   └── ultra_low_precision.js
│   ├── benchmark/             # Benchmarking components
│   │   ├── benchmark_runner.js
│   │   └── visualizer.js
│   ├── storage/               # Storage adapters
│   │   ├── indexed_db.js      # Browser storage
│   │   └── file_storage.js    # Node.js storage
│   ├── react/                 # React integration
│   │   ├── hooks.js           # React hooks
│   │   └── components.js      # React components
│   └── browser/               # Browser-specific code
│       ├── capabilities.js    # Browser capability detection
│       └── optimizations/     # Browser-specific optimizations
│           ├── firefox.js
│           ├── chrome.js
│           ├── edge.js
│           └── safari.js
├── dist/                      # Distribution files
│   ├── ipfs-accelerate.js     # UMD bundle
│   ├── ipfs-accelerate.min.js # Minified UMD bundle
│   └── ipfs-accelerate.esm.js # ESM bundle
├── examples/                  # Example implementations
│   ├── browser/               # Browser examples
│   │   ├── basic/             # Basic examples
│   │   ├── react/             # React examples
│   │   └── vue/               # Vue.js examples
│   └── node/                  # Node.js examples
├── test/                      # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── benchmarks/            # Benchmark tests
│   └── browser/               # Browser-specific tests
├── reports/                   # Generated reports
│   ├── benchmarks/            # Benchmark reports
│   ├── compatibility/         # Browser compatibility reports
│   └── performance/           # Performance reports
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── examples/              # Example documentation
│   └── guides/                # Usage guides
└── package.json               # NPM package definition
```

This structure mirrors the Python SDK while adapting for JavaScript and browser-specific needs.

### Key Features

- **Web Hardware Acceleration**: Automatic detection and utilization of WebNN, WebGPU, and WebAssembly
- **IPFS Integration**: Optimized IPFS content loading and distribution in the browser
- **P2P Optimization**: Enhanced content distribution through peer-to-peer network optimization
- **IndexedDB Integration**: Built-in storage and analysis of acceleration results
- **Cross-Browser Support**: Works across Chrome, Firefox, Edge, and Safari with appropriate fallbacks
- **Browser-Specific Optimizations**: Special optimizations for different browsers (e.g., Firefox for audio models)
- **Ultra-Low Precision Framework**: Advanced quantization support from 8-bit down to 2-bit precision
- **Shader Precompilation**: Improved startup times through shader precompilation
- **React Integration**: Dedicated React hooks for easy integration

### Architecture

The SDK consists of these core components:

1. **IPFS Integration Layer**: Interfaces with IPFS for content loading and storage.
2. **Web Hardware Acceleration Layer**: Detects and utilizes available web hardware (WebNN, WebGPU, WebAssembly).
3. **P2P Network Optimizer**: Optimizes content distribution across peers.
4. **IndexedDB Storage**: Stores and analyzes benchmark results.
5. **Configuration Manager**: Manages SDK settings and preferences.
6. **Model Registry**: Provides cross-browser model management and compatibility information.
7. **Benchmarking System**: Measures and analyzes performance across browsers.
8. **Quantization Engine**: Enables advanced precision control for models.
9. **Browser Detection**: Intelligent browser capability detection.
10. **WebSocket Bridge**: Communication with backend services.

## Installation

### Requirements

- Node.js 14.x or newer (for Node.js environment)
- Modern browser with WebNN/WebGPU support (Chrome 113+, Edge 113+, Firefox 113+, Safari 16.4+)
- IPFS companion or local node for P2P functionality (optional)

### Browser Installation

```html
<!-- Via CDN -->
<script src="https://cdn.jsdelivr.net/npm/ipfs-accelerate-js@0.4.0/dist/ipfs-accelerate.min.js"></script>

<!-- Or include locally after download -->
<script src="path/to/ipfs-accelerate.min.js"></script>

<!-- Module import -->
<script type="module">
  import { WebAccelerator, BrowserHardwareDetector } from 'https://cdn.jsdelivr.net/npm/ipfs-accelerate-js@0.4.0/dist/ipfs-accelerate.esm.js';
</script>
```

### NPM Installation (Node.js or bundled web apps)

```bash
# Install the package
npm install ipfs-accelerate-js

# Or with yarn
yarn add ipfs-accelerate-js
```

```javascript
// ES Module import
import { WebAccelerator, BrowserHardwareDetector } from 'ipfs-accelerate-js';

// CommonJS import
const { WebAccelerator, BrowserHardwareDetector } = require('ipfs-accelerate-js');
```

## Core Components

### WebAccelerator Class

The central class that handles acceleration in browser environments.

```javascript
import { WebAccelerator } from 'ipfs-accelerate-js';

// Create an accelerator instance
const accelerator = new WebAccelerator({
  autoDetectHardware: true,
  preferredBackend: 'webgpu',
  fallbackOrder: ['webnn', 'wasm'],
  enableP2P: true,
  storeResults: true
});

// Initialize the accelerator (detects hardware capabilities)
await accelerator.initialize();

// Get device capabilities
const capabilities = accelerator.getCapabilities();
console.log('WebGPU support:', capabilities.webgpu.supported);
console.log('WebNN support:', capabilities.webnn.supported);

// Load model from IPFS and accelerate inference
const result = await accelerator.accelerate({
  modelId: 'QmHash...',  // IPFS CID for model
  modelType: 'text',
  input: 'This is a test sentence'
});

console.log('Processing time:', result.processingTime, 'ms');
console.log('Using hardware:', result.hardware);
```

### BrowserHardwareDetector Class

Detects and reports on hardware capabilities in the browser environment.

```javascript
import { BrowserHardwareDetector } from 'ipfs-accelerate-js';

// Create detector
const detector = new BrowserHardwareDetector();

// Check overall hardware capabilities
const capabilities = await detector.detectCapabilities();
console.log('Hardware capabilities:', capabilities);

// Check if specific hardware is supported
const webgpuSupported = await detector.isSupported('webgpu');
const webnnSupported = await detector.isSupported('webnn');
console.log('WebGPU supported:', webgpuSupported);
console.log('WebNN supported:', webnnSupported);

// Get full details about specific backend
const webgpuDetails = await detector.getBackendDetails('webgpu');
console.log('WebGPU details:', webgpuDetails);

// Get optimal backend for a model type
const optimalBackend = await detector.getOptimalBackend('text');
console.log('Optimal backend for text models:', optimalBackend);

// Check if browser has real hardware or simulation
const isRealWebGPU = await detector.isRealHardware('webgpu');
console.log('Real WebGPU hardware:', isRealWebGPU);
```

### ModelManager Class

Manages models across different backends with a unified API.

```javascript
import { ModelManager } from 'ipfs-accelerate-js';

// Create a model manager
const modelManager = new ModelManager({
  storagePrefix: 'my-app',
  enableCaching: true
});

// Load a model (model will be loaded with optimal backend)
const model = await modelManager.loadModel('bert-base-uncased');

// Run inference
const embedding = await model.getEmbeddings('This is a test sentence');
console.log('Embedding length:', embedding.length);

// Switch backend if needed
await model.switchBackend('webnn');
const embeddingWebNN = await model.getEmbeddings('This is another test');

// Get model information
const modelInfo = model.getInfo();
console.log('Model info:', modelInfo);

// Unload the model to free resources
await model.unload();
```

### StorageManager Class

Manages result storage in IndexedDB for persistent benchmarking and analysis.

```javascript
import { StorageManager } from 'ipfs-accelerate-js';

// Create storage manager
const storage = new StorageManager({
  databaseName: 'acceleration-results',
  storageVersion: 1
});

// Initialize storage (create/open database)
await storage.initialize();

// Store a result
const resultId = await storage.storeAccelerationResult({
  modelName: 'bert-base-uncased',
  hardware: 'webgpu',
  latencyMs: 45.2,
  throughputItemsPerSecond: 22.1,
  memoryUsageMb: 125.3,
  browserInfo: navigator.userAgent,
  timestamp: new Date().toISOString()
});

// Query results
const results = await storage.getAccelerationResults({
  modelName: 'bert-base-uncased',
  limit: 10
});
console.log(`Found ${results.length} results`);

// Generate a report (returns HTML or Markdown)
const report = await storage.generateReport({
  format: 'html',
  includeCharts: true
});

// Export results to JSON
const jsonData = await storage.exportResults({
  format: 'json',
  modelNames: ['bert-base-uncased'],
  hardwareTypes: ['webgpu', 'webnn']
});
```

### BenchmarkRunner Class

Runs comprehensive benchmarks across backends and configurations.

```javascript
import { BenchmarkRunner, HardwareProfile } from 'ipfs-accelerate-js';

// Create benchmark runner
const benchmarkRunner = new BenchmarkRunner({
  modelIds: ['bert-base-uncased', 'vit-base', 'whisper-tiny'],
  hardwareProfiles: [
    new HardwareProfile({ backend: 'webgpu', browser: 'chrome' }),
    new HardwareProfile({ backend: 'webnn', browser: 'edge' }),
    new HardwareProfile({ backend: 'webgpu', browser: 'firefox' })
  ],
  metrics: ['latency', 'throughput', 'memory', 'power'],
  iterations: 50,
  warmupIterations: 5
});

// Run benchmarks
const { benchmarkId, results } = await benchmarkRunner.run();

// Generate visualization (returns HTML with interactive charts)
const visualization = await benchmarkRunner.visualize({
  containerId: 'benchmark-chart',
  title: 'Model Performance Across Browsers',
  metricToVisualize: 'latency',
  colorScheme: 'blue'
});

// Export results in various formats
await benchmarkRunner.exportResults({
  format: 'csv',
  filename: 'benchmark_results.csv'
});
```

### React Integration

The SDK includes dedicated React hooks for easy integration.

```jsx
import React, { useState } from 'react';
import { useModel, useHardwareInfo } from 'ipfs-accelerate-js/react';

function EmbeddingComponent() {
  // React hook for easy model loading
  const { model, status, error } = useModel({
    modelId: 'bert-base-uncased',
    autoHardwareSelection: true,
    fallbackOrder: ['webgpu', 'webnn', 'wasm']
  });

  // Hook for hardware information
  const { capabilities, isReady, optimalBackend } = useHardwareInfo();

  const [input, setInput] = useState('');
  const [embedding, setEmbedding] = useState(null);

  async function generateEmbedding() {
    if (model && input) {
      const result = await model.getEmbeddings(input);
      setEmbedding(result);
    }
  }

  return (
    <div>
      <div>
        <h3>Hardware Status</h3>
        {isReady ? (
          <ul>
            <li>WebGPU: {capabilities.webgpu.supported ? 'Yes' : 'No'}</li>
            <li>WebNN: {capabilities.webnn.supported ? 'Yes' : 'No'}</li>
            <li>Optimal backend: {optimalBackend}</li>
          </ul>
        ) : (
          <p>Detecting hardware capabilities...</p>
        )}
      </div>

      <div>
        <h3>Text Embedding</h3>
        <input 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          placeholder="Enter text to embed"
        />
        <button 
          onClick={generateEmbedding} 
          disabled={status !== 'loaded' || !input}
        >
          Generate Embedding
        </button>
        {status === 'loading' && <p>Loading model...</p>}
        {error && <p>Error: {error.message}</p>}
        {embedding && <p>Embedding generated: {embedding.length} dimensions</p>}
      </div>
    </div>
  );
}
```

### QuantizationEngine Class

Enables model quantization for faster inference and lower memory usage.

```javascript
import { QuantizationEngine } from 'ipfs-accelerate-js';

// Create quantization engine
const quantizer = new QuantizationEngine();

// Define calibration data
const calibrationTexts = [
  "This is a sample sentence for calibration.",
  "Machine learning models benefit from proper quantization.",
  "Multiple examples ensure representative distributions."
];

// Configure quantization
const quantizationConfig = {
  bits: 4,
  scheme: "symmetric",
  mixedPrecision: true,
  perChannel: true,
  layerExclusions: ["embeddings", "output_projection"],
  // WebGPU-specific optimizations
  shaderOptimizations: true,
  computeShaderPacking: true
};

// Quantize model
const quantizedModel = await quantizer.quantize({
  modelId: 'bert-base-uncased',
  calibrationData: calibrationTexts,
  quantizationConfig: quantizationConfig,
  targetBackend: 'webgpu'
});

// Run inference with quantized model
const embedding = await quantizedModel.getEmbeddings("This is a test");

// Compare performance with original model
const comparison = await quantizer.comparePerformance({
  originalModelId: 'bert-base-uncased',
  quantizedModel: quantizedModel,
  testInput: "This is a benchmark test",
  metrics: ['latency', 'memory', 'accuracy']
});

console.log('Performance comparison:', comparison);
```

## Usage Examples

### Basic Acceleration

The following examples demonstrate how to use the SDK's core acceleration features. These examples are located in the `examples/browser/basic` and `examples/node/basic` directories in the repo structure.

```javascript
// examples/browser/basic/text_acceleration.js
import { WebAccelerator } from 'ipfs-accelerate-js';

// Create accelerator
const accelerator = new WebAccelerator();
await accelerator.initialize();

// Text model
const textResult = await accelerator.accelerate({
  modelId: 'bert-base-uncased',
  modelType: 'text',
  input: 'This is a test of IPFS acceleration.',
  config: { autoSelectHardware: true }
});

console.log(`Processing time: ${textResult.processingTime} ms`);
console.log(`Using hardware: ${textResult.hardware}`);

// Vision model
const visionResult = await accelerator.accelerate({
  modelId: 'vit-base',
  modelType: 'vision',
  input: { imageUrl: 'https://example.com/test_image.jpg' },
  config: { hardware: 'webgpu' }  // Explicitly specify hardware
});

// Store results in appropriate location (Node.js environment)
if (typeof window === 'undefined') {
  // Node.js environment
  const fs = require('fs');
  const path = require('path');
  
  // Create reports directory if it doesn't exist
  const reportsDir = path.join(__dirname, '../../reports/benchmarks');
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }
  
  // Save results
  fs.writeFileSync(
    path.join(reportsDir, `acceleration_results_${new Date().toISOString().slice(0,10)}.json`),
    JSON.stringify({ textResult, visionResult }, null, 2)
  );
} else {
  // Browser environment - use IndexedDB storage
  const storage = new StorageManager({ databaseName: 'acceleration-results' });
  await storage.initialize();
  await storage.storeAccelerationResult(textResult);
  await storage.storeAccelerationResult(visionResult);
}
```

### Advanced Configuration

The following example demonstrates advanced configuration options and would typically be found in `examples/browser/advanced/audio_optimization.js` or similar files in the repo structure.

```javascript
// examples/browser/advanced/firefox_audio_optimization.js
import { WebAccelerator } from 'ipfs-accelerate-js';
import { getOptimalConfig } from 'ipfs-accelerate-js/browser/optimizations';

// Create accelerator with detailed configuration
const accelerator = new WebAccelerator({
  enableLogging: true,
  logLevel: 'verbose',
  enableP2P: true,
  reportDir: '../../reports'  // Used in Node.js environments
});

await accelerator.initialize();

// Get browser-specific optimizations (from src/browser/optimizations/firefox.js)
const browserOptimizations = await getOptimalConfig({
  browser: 'firefox',
  modelType: 'audio'
});

// Advanced configuration for audio model
const result = await accelerator.accelerate({
  modelId: 'whisper-tiny',
  modelType: 'audio',
  input: { audioUrl: 'https://example.com/test_audio.mp3' },
  config: {
    hardware: 'webgpu',         // Use WebGPU
    browser: 'firefox',         // Optimized for Firefox
    precision: 8,               // Use 8-bit precision
    mixedPrecision: true,       // Use mixed precision
    useFirefoxOptimizations: true, // Use Firefox audio optimizations
    computeShaders: true,       // Use compute shaders
    shaderPrecompilation: true, // Precompile shaders
    p2pOptimization: true,      // Use P2P optimization
    storeResults: true,         // Store results in IndexedDB
    memoryLimit: 2048,          // Set memory limit in MB
    ...browserOptimizations     // Include browser-specific optimizations
  }
});

// Generate performance report and save to appropriate location
const report = await accelerator.generateReport({
  modelId: 'whisper-tiny',
  testData: result,
  format: 'html',
  reportType: 'performance',
  // The following is used in Node.js environments
  outputDir: '../../reports/performance/audio_models/firefox',
  filename: `whisper_tiny_firefox_${new Date().toISOString().slice(0,10)}.html`
});

if (typeof window !== 'undefined') {
  // Browser environment - display report
  document.getElementById('report-container').innerHTML = report;
  
  // Option to download report
  const blob = new Blob([report], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `whisper_tiny_firefox_${new Date().toISOString().slice(0,10)}.html`;
  a.click();
}
```

### Cross-Browser Testing

```javascript
import { WebAccelerator, BrowserHardwareDetector } from 'ipfs-accelerate-js';

async function runCrossBrowserTest() {
  // Test is being run in an automated browser testing framework
  // that will execute this code in multiple browsers
  
  // Detect current browser
  const detector = new BrowserHardwareDetector();
  const capabilities = await detector.detectCapabilities();
  const browserName = capabilities.browserName;
  
  console.log(`Testing in ${browserName}`);
  
  // Initialize accelerator
  const accelerator = new WebAccelerator();
  await accelerator.initialize();
  
  // Run test
  try {
    const result = await accelerator.accelerate({
      modelId: 'bert-base-uncased',
      modelType: 'text',
      input: 'This is a cross-browser test.',
      config: { 
        // Use optimal hardware for current browser
        autoSelectHardware: true
      }
    });
    
    console.log(`${browserName} test result:`, {
      latency: result.processingTime,
      throughput: result.throughput,
      hardware: result.hardware,
      memoryUsage: result.memoryUsage
    });
    
    return result;
  } catch (error) {
    console.error(`${browserName} test failed:`, error);
    return { error: error.message, browser: browserName };
  }
}
```

### Browser-Specific Optimizations

```javascript
import { WebAccelerator, BrowserHardwareDetector } from 'ipfs-accelerate-js';

async function testBrowserOptimizations() {
  const detector = new BrowserHardwareDetector();
  const currentBrowser = (await detector.detectCapabilities()).browserName;
  
  // Initialize accelerator
  const accelerator = new WebAccelerator();
  await accelerator.initialize();
  
  let results = {};
  
  // Test with Firefox audio optimizations
  if (currentBrowser === 'firefox') {
    // Firefox has optimized compute shaders for audio models
    const firefoxResult = await accelerator.accelerate({
      modelId: 'whisper-tiny',
      modelType: 'audio',
      input: { audioUrl: 'https://example.com/test_audio.mp3' },
      config: {
        hardware: 'webgpu',
        useFirefoxOptimizations: true,
        computeShaders: true
      }
    });
    
    results.firefox = {
      throughput: firefoxResult.throughput,
      processingTime: firefoxResult.processingTime
    };
  }
  
  // Test with Edge WebNN optimizations
  if (currentBrowser === 'edge') {
    // Edge has best WebNN support
    const edgeResult = await accelerator.accelerate({
      modelId: 'bert-base-uncased',
      modelType: 'text',
      input: 'This is a test sentence.',
      config: {
        hardware: 'webnn'
      }
    });
    
    results.edge = {
      throughput: edgeResult.throughput,
      processingTime: edgeResult.processingTime
    };
  }
  
  // Test with Chrome shader precompilation
  if (currentBrowser === 'chrome') {
    // Chrome has good shader precompilation
    const chromeResult = await accelerator.accelerate({
      modelId: 'vit-base',
      modelType: 'vision',
      input: { imageUrl: 'https://example.com/test_image.jpg' },
      config: {
        hardware: 'webgpu',
        shaderPrecompilation: true
      }
    });
    
    results.chrome = {
      throughput: chromeResult.throughput,
      processingTime: chromeResult.processingTime,
      shaderCompilationTime: chromeResult.shaderCompilationTime
    };
  }
  
  return results;
}
```

### IndexedDB Storage and Analysis

The following example demonstrates comprehensive storage and analysis functionality, typically located in the `examples/browser/advanced/` directory.

```javascript
// examples/browser/advanced/benchmark_storage_analysis.js
import { WebAccelerator, StorageManager } from 'ipfs-accelerate-js';
import { visualizeBenchmarks } from 'ipfs-accelerate-js/src/benchmark/visualizer';

async function runAndStoreTests() {
  // Initialize storage using the StorageManager from src/storage/indexed_db.js
  const storage = new StorageManager({ 
    databaseName: 'acceleration-results',
    storageVersion: 1
  });
  await storage.initialize();
  
  // Initialize accelerator with storage integration
  const accelerator = new WebAccelerator({
    storage: storage,
    storeResults: true,
    // Path for Node.js environments, ignored in browser
    reportDir: path.join(__dirname, '../../reports')
  });
  await accelerator.initialize();
  
  // Run tests for multiple backends
  const backends = ['webgpu', 'webnn', 'wasm'];
  const modelId = 'bert-base-uncased';
  const input = 'This is a test for database analysis.';
  const results = {};
  
  for (const backend of backends) {
    if (await accelerator.isBackendSupported(backend)) {
      // Run acceleration
      const result = await accelerator.accelerate({
        modelId,
        modelType: 'text',
        input,
        config: { hardware: backend }
      });
      
      results[backend] = result;
      console.log(`Tested ${backend}: ${result.processingTime} ms`);
    } else {
      console.log(`Backend ${backend} not supported in this browser`);
    }
  }
  
  // Generate report with appropriate paths for environment
  const reportOptions = {
    format: 'html',
    title: 'Browser Backend Comparison',
    includeCharts: true
  };
  
  // Add environment-specific options
  if (typeof window === 'undefined') {
    // Node.js environment - set output directory
    reportOptions.outputDir = path.join(__dirname, '../../reports/benchmarks/text_models');
    reportOptions.filename = `bert_backend_comparison_${new Date().toISOString().slice(0,10)}.html`;
  }
  
  const report = await storage.generateReport(reportOptions);
  
  // For browser environments
  if (typeof window !== 'undefined') {
    // Display the report in browser
    document.getElementById('report-container').innerHTML = report;
    
    // Create interactive visualization in browser
    const vizElement = document.getElementById('visualization-container');
    await visualizeBenchmarks({
      data: results,
      element: vizElement,
      title: 'Backend Performance Comparison',
      metrics: ['processingTime', 'throughput', 'memoryUsage']
    });
  }
  
  // Get aggregated statistics with file saving option for Node.js
  const statsOptions = {
    groupBy: 'hardware',
    metrics: ['avg_latency', 'throughput']
  };
  
  // Add file options for Node.js
  if (typeof window === 'undefined') {
    statsOptions.saveToFile = true;
    statsOptions.outputPath = path.join(
      __dirname, 
      '../../reports/benchmarks/stats',
      `bert_stats_${new Date().toISOString().slice(0,10)}.json`
    );
  }
  
  const stats = await storage.getAggregatedStats(statsOptions);
  console.log('Performance statistics:', stats);
  
  // Export raw data
  await storage.exportResults({
    format: 'json',
    modelNames: [modelId],
    // The following is only used in Node.js
    filename: `bert_data_${new Date().toISOString().slice(0,10)}.json`,
    outputDir: path.join(__dirname, '../../reports/benchmarks/raw_data')
  });
}
```

### Integration with React Application

```jsx
import React, { useState } from 'react';
import { useModel, useBrowserCapabilities, useP2PStatus } from 'ipfs-accelerate-js/react';

function ModelDashboard() {
  // Model loading hook
  const { model, status, error, switchBackend } = useModel({
    modelId: 'bert-base-uncased',
    autoLoad: true,
    fallbackOrder: ['webgpu', 'webnn', 'wasm']
  });
  
  // Browser capabilities hook
  const { capabilities, isDetecting, optimalBackend } = useBrowserCapabilities();
  
  // P2P network status hook
  const { isEnabled, peerCount, networkHealth, enableP2P, disableP2P } = useP2PStatus();
  
  // UI state
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  const [backend, setBackend] = useState(null);
  
  // Run inference
  async function runInference() {
    if (model && input) {
      const startTime = performance.now();
      const embedding = await model.getEmbeddings(input);
      const endTime = performance.now();
      
      setResult({
        embedding: embedding,
        dimensions: embedding.length,
        processingTime: (endTime - startTime).toFixed(2),
        backend: model.currentBackend
      });
    }
  }
  
  // Switch backend
  async function handleBackendSwitch(newBackend) {
    await switchBackend(newBackend);
    setBackend(newBackend);
  }
  
  return (
    <div className="dashboard">
      <div className="capabilities-panel">
        <h2>Browser Capabilities</h2>
        {isDetecting ? (
          <p>Detecting capabilities...</p>
        ) : (
          <ul>
            <li>WebGPU: {capabilities.webgpu.supported ? '✅' : '❌'}</li>
            <li>WebNN: {capabilities.webnn.supported ? '✅' : '❌'}</li>
            <li>WebAssembly: {capabilities.wasm.supported ? '✅' : '❌'}</li>
            <li>Optimal backend: {optimalBackend}</li>
          </ul>
        )}
      </div>
      
      <div className="p2p-panel">
        <h2>P2P Network</h2>
        <div className="status">
          Status: {isEnabled ? '✅ Enabled' : '❌ Disabled'}
          {isEnabled && (
            <>
              <p>Peers: {peerCount}</p>
              <p>Network Health: {networkHealth}</p>
            </>
          )}
        </div>
        <div className="controls">
          <button onClick={enableP2P} disabled={isEnabled}>Enable P2P</button>
          <button onClick={disableP2P} disabled={!isEnabled}>Disable P2P</button>
        </div>
      </div>
      
      <div className="inference-panel">
        <h2>Text Embedding</h2>
        <div className="backend-selector">
          <button onClick={() => handleBackendSwitch('webgpu')} disabled={!capabilities.webgpu.supported}>WebGPU</button>
          <button onClick={() => handleBackendSwitch('webnn')} disabled={!capabilities.webnn.supported}>WebNN</button>
          <button onClick={() => handleBackendSwitch('wasm')}>WebAssembly</button>
        </div>
        <textarea 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          placeholder="Enter text to embed"
        />
        <button 
          onClick={runInference} 
          disabled={status !== 'loaded' || !input}
        >
          Run Inference
        </button>
        {status === 'loading' && <p>Loading model...</p>}
        {error && <p className="error">Error: {error.message}</p>}
        
        {result && (
          <div className="result">
            <h3>Result</h3>
            <p>Dimensions: {result.dimensions}</p>
            <p>Processing Time: {result.processingTime} ms</p>
            <p>Backend Used: {result.backend}</p>
          </div>
        )}
      </div>
    </div>
  );
}
```

## API Reference

### WebAccelerator Class

```javascript
class WebAccelerator {
  constructor(options: {
    autoDetectHardware?: boolean,
    preferredBackend?: 'webgpu' | 'webnn' | 'wasm',
    fallbackOrder?: Array<'webgpu' | 'webnn' | 'wasm'>,
    enableP2P?: boolean,
    storeResults?: boolean,
    storage?: StorageManager,
    logLevel?: 'verbose' | 'info' | 'error' | 'none'
  })
  
  async initialize(): Promise<void>
  
  async accelerate({
    modelId: string,
    modelType: 'text' | 'vision' | 'audio' | 'multimodal',
    input: any,
    config?: {
      hardware?: 'webgpu' | 'webnn' | 'wasm',
      browser?: 'chrome' | 'firefox' | 'edge' | 'safari',
      precision?: number,
      mixedPrecision?: boolean,
      useFirefoxOptimizations?: boolean,
      computeShaders?: boolean,
      shaderPrecompilation?: boolean,
      p2pOptimization?: boolean,
      storeResults?: boolean,
      memoryLimit?: number
    }
  }): Promise<{
    status: string,
    processingTime: number,
    throughput: number,
    hardware: string,
    memoryUsage: number,
    result: any
  }>
  
  getCapabilities(): Object
  
  async isBackendSupported(backend: string): Promise<boolean>
  
  async getOptimalBackend(modelType: string): Promise<string>
}
```

### BrowserHardwareDetector Class

```javascript
class BrowserHardwareDetector {
  constructor(options?: {
    useCache?: boolean,
    cacheExpiryMs?: number
  })
  
  async detectCapabilities(): Promise<{
    browserName: string,
    browserVersion: string,
    osName: string,
    osVersion: string,
    deviceType: 'desktop' | 'mobile' | 'tablet' | 'unknown',
    webgpu: {
      supported: boolean,
      adapter: string,
      features: string[]
    },
    webnn: {
      supported: boolean,
      backends: string[]
    },
    wasm: {
      supported: boolean,
      threads: boolean,
      simd: boolean
    }
  }>
  
  async isSupported(backend: 'webgpu' | 'webnn' | 'wasm'): Promise<boolean>
  
  async getBackendDetails(backend: string): Promise<Object>
  
  async getOptimalBackend(modelType: string): Promise<string>
  
  async isRealHardware(backend: string): Promise<boolean>
}
```

### ModelManager Class

```javascript
class ModelManager {
  constructor(options?: {
    storagePrefix?: string,
    enableCaching?: boolean,
    cacheSizeLimit?: number
  })
  
  async loadModel(modelId: string, options?: {
    backend?: 'webgpu' | 'webnn' | 'wasm',
    precision?: number,
    mixedPrecision?: boolean
  }): Promise<ModelInterface>
  
  async unloadModel(modelId: string): Promise<boolean>
  
  getLoadedModels(): string[]
  
  clearCache(): Promise<void>
}

interface ModelInterface {
  modelId: string
  currentBackend: string
  
  async getEmbeddings(text: string): Promise<number[]>
  
  async predict(input: any): Promise<any>
  
  async switchBackend(newBackend: string): Promise<boolean>
  
  getInfo(): Object
  
  async unload(): Promise<void>
}
```

### StorageManager Class

```javascript
class StorageManager {
  constructor(options?: {
    databaseName?: string,
    storageVersion?: number
  })
  
  async initialize(): Promise<void>
  
  async storeAccelerationResult(result: Object): Promise<string>
  
  async getAccelerationResults(options?: {
    modelName?: string,
    hardware?: string,
    limit?: number,
    offset?: number,
    startDate?: string,
    endDate?: string
  }): Promise<Object[]>
  
  async generateReport(options?: {
    format?: 'html' | 'markdown' | 'json',
    title?: string,
    includeCharts?: boolean,
    groupBy?: string,
    outputPath?: string,      // Path for Node.js environments
    outputDir?: string,       // Directory for Node.js environments
    reportType?: 'benchmark' | 'performance' | 'compatibility',
    browserFilter?: string[]  // Filter by browser types
  }): Promise<string>
  
  async exportResults(options?: {
    format?: 'json' | 'csv',
    modelNames?: string[],
    hardwareTypes?: string[],
    startDate?: string,
    endDate?: string,
    filename?: string,       // For browser downloads or Node.js file writing
    outputDir?: string       // Directory path for Node.js environments
  }): Promise<any>
  
  async getAggregatedStats(options?: {
    groupBy?: 'hardware' | 'model' | 'browser',
    metrics?: string[],
    saveToFile?: boolean,     // Save to file (Node.js)
    outputPath?: string       // Output path for saved stats
  }): Promise<Object>
  
  async close(): Promise<void>
}
```

### BenchmarkRunner Class

```javascript
class BenchmarkRunner {
  constructor(options: {
    modelIds: string[],
    hardwareProfiles: HardwareProfile[],
    metrics?: string[],
    iterations?: number,
    warmupIterations?: number,
    batchSizes?: number[],
    sequenceLengths?: number[],
    options?: Object
  })
  
  async run(): Promise<{
    benchmarkId: string,
    results: Object
  }>
  
  async visualize(options?: {
    containerId?: string,
    title?: string,
    metricToVisualize?: string,
    colorScheme?: string
  }): Promise<string>
  
  async exportResults(options?: {
    format?: 'json' | 'csv' | 'html',
    filename?: string,
    outputDir?: string,       // Directory for storing reports
    browserTypes?: string[],  // Filter by browser types
    modelTypes?: string[],    // Filter by model types
    reportCategory?: 'benchmarks' | 'performance' | 'compatibility'
  }): Promise<any>
}

class HardwareProfile {
  constructor(options: {
    backend: 'webgpu' | 'webnn' | 'wasm',
    browser?: 'chrome' | 'firefox' | 'edge' | 'safari',
    precision?: number,
    optimizationLevel?: 'default' | 'performance' | 'size',
    features?: Object
  })
}
```

### QuantizationEngine Class

```javascript
class QuantizationEngine {
  constructor(options?: {
    useCache?: boolean
  })
  
  async quantize(options: {
    modelId: string,
    calibrationData: any[],
    quantizationConfig: {
      bits: number,
      scheme: 'symmetric' | 'asymmetric',
      mixedPrecision?: boolean,
      perChannel?: boolean,
      layerExclusions?: string[],
      shaderOptimizations?: boolean,
      computeShaderPacking?: boolean
    },
    targetBackend?: 'webgpu' | 'webnn' | 'wasm'
  }): Promise<ModelInterface>
  
  async comparePerformance(options: {
    originalModelId: string,
    quantizedModel: ModelInterface,
    testInput: any,
    metrics?: string[]
  }): Promise<Object>
}
```

### React Hooks

```javascript
// Model loading hook
function useModel(options: {
  modelId: string,
  autoLoad?: boolean,
  autoHardwareSelection?: boolean,
  fallbackOrder?: string[]
}): {
  model: ModelInterface | null,
  status: 'loading' | 'loaded' | 'error' | 'idle',
  error: Error | null,
  switchBackend: (newBackend: string) => Promise<boolean>
}

// Browser capabilities hook
function useBrowserCapabilities(): {
  capabilities: Object,
  isDetecting: boolean,
  optimalBackend: string | null
}

// P2P network status hook
function useP2PStatus(): {
  isEnabled: boolean,
  peerCount: number,
  networkHealth: number,
  enableP2P: () => Promise<void>,
  disableP2P: () => Promise<void>
}

// Acceleration hook
function useAcceleration(options: {
  modelId: string,
  modelType: string,
  backend?: string,
  autoInitialize?: boolean
}): {
  accelerate: (input: any, config?: Object) => Promise<Object>,
  isReady: boolean,
  error: Error | null,
  capabilities: Object
}
```

## Best Practices

1. **Browser Detection and Hardware Selection**:
   - Always detect browser capabilities before attempting to use acceleration
   - Provide fallbacks in preferred order (e.g., WebGPU → WebNN → WebAssembly)
   - Consider browser-specific optimizations (Firefox for audio, Edge for WebNN)

2. **Performance Optimization**:
   - Use shader precompilation to improve startup time
   - Consider quantization for memory-constrained environments
   - Enable compute shader optimizations for audio models
   - Set appropriate memory limits for the target device

3. **Resource Management**:
   - Unload models when they are no longer needed
   - Clear cache periodically to avoid excessive storage usage
   - Close IndexedDB connections when finished
   - Be aware of WebGPU adapter resource limits

4. **Error Handling**:
   - Always use try-catch blocks when working with hardware acceleration
   - Provide clear fallback paths for unsupported features
   - Monitor for resource exhaustion errors, especially on mobile devices
   - Handle WebGPU context loss gracefully

5. **React Integration**:
   - Use the provided hooks for cleaner integration
   - Avoid multiple initializations of the same model
   - Use the useEffect cleanup function to properly unload models

## Troubleshooting

The SDK provides comprehensive troubleshooting utilities located in various parts of the codebase. These examples would typically be found in the `examples/browser/troubleshooting/` directory.

### Common Issues

1. **WebGPU Support Issues**:
   ```javascript
   // examples/browser/troubleshooting/webgpu_diagnostics.js
   import { BrowserHardwareDetector } from 'ipfs-accelerate-js';
   import { createDiagnosticReport } from 'ipfs-accelerate-js/src/browser/capabilities';
   
   async function diagnoseWebGPU() {
     // Use BrowserHardwareDetector from src/browser/capabilities.js
     const detector = new BrowserHardwareDetector();
     const capabilities = await detector.detectCapabilities();
     
     if (!capabilities.webgpu.supported) {
       console.error('WebGPU not supported in this browser');
       return;
     }
     
     const adapterDetails = await detector.getBackendDetails('webgpu');
     console.log('WebGPU adapter:', adapterDetails.adapter);
     console.log('WebGPU features:', adapterDetails.features);
     console.log('WebGPU limits:', adapterDetails.limits);
     
     // Generate full diagnostic report with appropriate paths
     const reportData = await createDiagnosticReport({
       hardwareType: 'webgpu',
       includeExtendedInfo: true,
       includeShaderTests: true
     });
     
     // Save report in appropriate location
     if (typeof window === 'undefined') {
       // Node.js environment
       const fs = require('fs');
       const path = require('path');
       const reportDir = path.join(__dirname, '../../reports/compatibility/feature_matrix');
       if (!fs.existsSync(reportDir)) {
         fs.mkdirSync(reportDir, { recursive: true });
       }
       
       fs.writeFileSync(
         path.join(reportDir, `webgpu_diagnostics_${new Date().toISOString().slice(0,10)}.json`),
         JSON.stringify(reportData, null, 2)
       );
     } else {
       // Browser environment
       console.log('Diagnostic report:', reportData);
       
       // Option to download the report
       const reportStr = JSON.stringify(reportData, null, 2);
       const blob = new Blob([reportStr], { type: 'application/json' });
       const url = URL.createObjectURL(blob);
       const a = document.createElement('a');
       a.href = url;
       a.download = `webgpu_diagnostics_${new Date().toISOString().slice(0,10)}.json`;
       a.click();
     }
   }
   ```

2. **Shader Compilation Errors**:
   ```javascript
   // examples/browser/troubleshooting/shader_compilation_diagnostics.js
   import { WebAccelerator } from 'ipfs-accelerate-js';
   import { generateShaderReport } from 'ipfs-accelerate-js/src/browser/optimizations/shader_diagnostics';
   
   async function diagnoseShaderIssues() {
     // Enable diagnostic mode for WebGPU
     const accelerator = new WebAccelerator({
       logLevel: 'verbose',
       diagnosticMode: true,
       // Path for Node.js environments
       reportDir: typeof window === 'undefined' ? 
         path.join(__dirname, '../../reports/compatibility') : undefined
     });
     
     await accelerator.initialize();
     
     try {
       const result = await accelerator.accelerate({
         modelId: 'bert-base-uncased',
         modelType: 'text',
         input: 'Test input',
         config: {
           hardware: 'webgpu',
           shaderCompilationDiagnostics: true
         }
       });
       
       // Generate shader compilation report
       const compilationReport = await generateShaderReport({
         modelId: 'bert-base-uncased',
         browser: result.browser,
         adapter: result.adapter,
         shaderMetrics: result.shaderMetrics
       });
       
       // Save/display report
       if (typeof window === 'undefined') {
         // Node.js environment
         const fs = require('fs');
         const path = require('path');
         const reportDir = path.join(__dirname, '../../reports/compatibility/shader_compilation');
         if (!fs.existsSync(reportDir)) {
           fs.mkdirSync(reportDir, { recursive: true });
         }
         
         fs.writeFileSync(
           path.join(reportDir, `shader_compilation_${new Date().toISOString().slice(0,10)}.json`),
           JSON.stringify(compilationReport, null, 2)
         );
       } else {
         // Browser environment
         console.log('Shader compilation report:', compilationReport);
         document.getElementById('shader-report').textContent = 
           JSON.stringify(compilationReport, null, 2);
       }
     } catch (error) {
       console.error('Shader compilation error:', error);
       
       // Create detailed error report
       const errorReport = {
         timestamp: new Date().toISOString(),
         browser: navigator?.userAgent,
         error: error.message,
         stack: error.stack,
         details: error.details || 'No additional details'
       };
       
       // Save/display error report
       if (typeof window === 'undefined') {
         // Node.js environment
         const fs = require('fs');
         const path = require('path');
         const errorDir = path.join(__dirname, '../../reports/compatibility/errors');
         if (!fs.existsSync(errorDir)) {
           fs.mkdirSync(errorDir, { recursive: true });
         }
         
         fs.writeFileSync(
           path.join(errorDir, `shader_error_${new Date().toISOString().slice(0,10)}.json`),
           JSON.stringify(errorReport, null, 2)
         );
       } else {
         // Browser environment
         console.error('Detailed shader error:', errorReport);
       }
     }
   }
   ```

3. **Memory Issues**:
   ```javascript
   // Set memory limits and monitor usage
   import { WebAccelerator } from 'ipfs-accelerate-js';
   
   const accelerator = new WebAccelerator();
   await accelerator.initialize();
   
   // Enable memory monitoring
   accelerator.enableMemoryMonitoring();
   
   // Run with memory limits
   const result = await accelerator.accelerate({
     modelId: 'bert-base-uncased',
     modelType: 'text',
     input: 'Test input',
     config: {
       memoryLimit: 512, // MB
       reportMemoryUsage: true
     }
   });
   
   console.log('Peak memory usage:', result.peakMemoryUsage, 'MB');
   console.log('Current memory usage:', result.currentMemoryUsage, 'MB');
   
   // Get memory usage history
   const memoryHistory = accelerator.getMemoryUsageHistory();
   console.log('Memory usage over time:', memoryHistory);
   ```

4. **Browser Compatibility Issues**:
   ```javascript
   // Detect browser and set optimal configuration
   import { BrowserHardwareDetector, WebAccelerator } from 'ipfs-accelerate-js';
   
   async function setupBrowserOptimalConfig() {
     const detector = new BrowserHardwareDetector();
     const browser = (await detector.detectCapabilities()).browserName;
     
     const accelerator = new WebAccelerator();
     await accelerator.initialize();
     
     let config = {};
     
     switch (browser) {
       case 'firefox':
         config = {
           hardware: 'webgpu',
           useFirefoxOptimizations: true,
           computeShaders: true
         };
         break;
       case 'edge':
         config = {
           hardware: 'webnn'
         };
         break;
       case 'chrome':
         config = {
           hardware: 'webgpu',
           shaderPrecompilation: true
         };
         break;
       default:
         config = {
           hardware: 'wasm',
           threads: 4
         };
     }
     
     return { accelerator, config };
   }
   ```

## Advanced Topics

This section covers advanced usage patterns located in the `examples/advanced/` directory, which demonstrates specialized SDK capabilities.

### Custom WebGPU Pipeline

You can implement custom WebGPU acceleration pipelines for specialized needs in the `src/worker/webgpu/custom_pipelines/` directory:

```javascript
// src/worker/webgpu/custom_pipelines/vector_multiply.js
import { WebGPUBackend } from 'ipfs-accelerate-js/src/hardware/backends/webgpu_backend';

class CustomWebGPUPipeline {
  constructor() {
    this.backend = new WebGPUBackend();
  }
  
  async initialize() {
    await this.backend.initialize();
    
    // Get device
    const device = this.backend.getDevice();
    
    // Create custom compute pipeline
    const shaderModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= arrayLength(&input)) {
            return;
          }
          
          // Custom computation
          output[idx] = input[idx] * 2.0;
        }
      `
    });
    
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
  }
  
  async process(inputData) {
    const device = this.backend.getDevice();
    
    // Create input buffer
    const inputBuffer = device.createBuffer({
      size: inputData.length * 4, // f32 = 4 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    // Create output buffer
    const outputBuffer = device.createBuffer({
      size: inputData.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    // Create staging buffer for reading results
    const stagingBuffer = device.createBuffer({
      size: inputData.length * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    
    // Write data to input buffer
    device.queue.writeBuffer(inputBuffer, 0, new Float32Array(inputData));
    
    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } }
      ]
    });
    
    // Create and submit command encoder
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(inputData.length / 64));
    computePass.end();
    
    // Copy output to staging buffer
    commandEncoder.copyBufferToBuffer(
      outputBuffer, 0,
      stagingBuffer, 0,
      inputData.length * 4
    );
    
    // Submit commands
    device.queue.submit([commandEncoder.finish()]);
    
    // Read result
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange());
    const output = Array.from(result);
    stagingBuffer.unmap();
    
    // Generate performance report if requested
    if (this.options?.generateReport) {
      // Create a report directory in the appropriate location
      if (typeof window === 'undefined') {
        const fs = require('fs');
        const path = require('path');
        const reportDir = path.join(__dirname, '../../../../reports/performance/webgpu');
        if (!fs.existsSync(reportDir)) {
          fs.mkdirSync(reportDir, { recursive: true });
        }
        
        // Write performance report
        const performanceReport = {
          timestamp: new Date().toISOString(),
          inputSize: inputData.length,
          outputSize: output.length,
          executionTime: performance.now() - this.startTime,
          adapter: this.backend.getAdapterInfo(),
          workgroupSize: 64,
          workgroups: Math.ceil(inputData.length / 64)
        };
        
        fs.writeFileSync(
          path.join(reportDir, `custom_pipeline_${new Date().toISOString().slice(0,10)}.json`),
          JSON.stringify(performanceReport, null, 2)
        );
      }
    }
    
    return output;
  }
  
  dispose() {
    // Clean up resources
    this.pipeline = null;
    this.backend.dispose();
  }
}
```

### Distributed Model Execution

The SDK supports distributed execution of large models across multiple tabs or workers. This functionality is implemented in the `src/advanced/model_sharding/` directory:

```javascript
// examples/browser/advanced/distributed_model_execution.js
import { ModelShardingManager } from 'ipfs-accelerate-js/src/advanced/model_sharding';
import { createDistributedReport } from 'ipfs-accelerate-js/src/benchmark/distributed_reporter';

async function runDistributedModel() {
  // Create sharding manager, implemented in src/advanced/model_sharding/index.js
  const shardingManager = new ModelShardingManager({
    modelId: 'llama-7b',
    numShards: 4,
    shardType: 'layer',
    reportMetrics: true, // Enable performance reporting
    reportDir: typeof window === 'undefined' ? 
      path.join(__dirname, '../../reports/performance/distributed') : undefined
  });
  
  // Initialize sharding (opens browser tabs or workers)
  await shardingManager.initialize();
  
  // Check initialization status
  const status = shardingManager.getStatus();
  console.log('Sharding status:', status);
  
  // Run inference across shards
  const startTime = performance.now();
  const result = await shardingManager.runInference({
    input: 'Write a short poem about WebGPU',
    maxTokens: 100,
    temperature: 0.7
  });
  const endTime = performance.now();
  
  console.log('Generated text:', result.text);
  console.log('Processing time:', result.processingTime, 'ms');
  
  // Generate performance report for distributed execution
  const distributedReport = await createDistributedReport({
    modelId: 'llama-7b',
    numShards: 4,
    shardType: 'layer',
    executionTime: endTime - startTime,
    shardMetrics: result.shardMetrics || [],
    memoryUsage: result.memoryUsage || {},
    outputLength: result.text?.length || 0
  });
  
  // Save or display the report
  if (typeof window === 'undefined') {
    // Node.js environment
    const fs = require('fs');
    const path = require('path');
    const reportDir = path.join(__dirname, '../../reports/performance/distributed');
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }
    
    fs.writeFileSync(
      path.join(reportDir, `distributed_model_${new Date().toISOString().slice(0,10)}.json`),
      JSON.stringify(distributedReport, null, 2)
    );
  } else {
    // Browser environment
    console.log('Distributed execution report:', distributedReport);
    
    // Option to download the report
    const reportStr = JSON.stringify(distributedReport, null, 2);
    const blob = new Blob([reportStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `distributed_model_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    
    // Display visualization if visualization container exists
    if (document.getElementById('distributed-visualization')) {
      // Import the visualizer (dynamically to avoid issues in Node.js)
      import('ipfs-accelerate-js/src/benchmark/visualizer').then(({ visualizeDistributedExecution }) => {
        visualizeDistributedExecution({
          data: distributedReport,
          element: document.getElementById('distributed-visualization')
        });
      });
    }
  }
  
  // Close all shards when done
  await shardingManager.dispose();
}
```

### Custom Model Cache

Implement a custom model cache for advanced storage strategies in the `src/model/cache/` directory:

```javascript
// src/model/cache/custom_cache.js
import { ModelCache } from 'ipfs-accelerate-js/src/model/cache/base_cache';
import { logManager } from 'ipfs-accelerate-js/src/utils/log_manager';

// Create logger for this module
const logger = logManager.getLogger('CustomModelCache');

class CustomModelCache extends ModelCache {
  constructor(options) {
    super(options);
    this.customStorage = new Map();
    this.cacheDir = options?.cacheDir;
    this.logCacheMetrics = options?.logCacheMetrics || false;
    
    // For Node.js environments, set cache directory
    if (typeof window === 'undefined' && this.cacheDir) {
      try {
        const fs = require('fs');
        const path = require('path');
        if (!fs.existsSync(this.cacheDir)) {
          fs.mkdirSync(this.cacheDir, { recursive: true });
        }
        logger.debug(`Cache directory set to: ${this.cacheDir}`);
      } catch (err) {
        logger.error(`Failed to create cache directory: ${err.message}`);
      }
    }
  }
  
  async initialize() {
    // Initialize custom storage
    try {
      if (typeof window !== 'undefined') {
        // Browser environment - use localStorage
        const cachedState = localStorage.getItem('model-cache-state');
        if (cachedState) {
          this.cacheState = JSON.parse(cachedState);
        } else {
          this.cacheState = {};
        }
      } else {
        // Node.js environment - use filesystem
        if (this.cacheDir) {
          const fs = require('fs');
          const path = require('path');
          const cacheStatePath = path.join(this.cacheDir, 'cache_state.json');
          
          if (fs.existsSync(cacheStatePath)) {
            const data = fs.readFileSync(cacheStatePath, 'utf8');
            this.cacheState = JSON.parse(data);
          } else {
            this.cacheState = {};
            fs.writeFileSync(cacheStatePath, JSON.stringify(this.cacheState), 'utf8');
          }
        } else {
          this.cacheState = {};
        }
      }
      
      this.initialized = true;
      if (this.logCacheMetrics) {
        await this._logCacheMetrics();
      }
    } catch (error) {
      logger.error(`Failed to initialize custom cache: ${error.message}`);
      throw error;
    }
  }
  
  async storeModel(modelId, modelData, metadata) {
    try {
      // Store in memory
      this.customStorage.set(modelId, {
        data: modelData,
        metadata: {
          ...metadata,
          timestamp: Date.now()
        }
      });
      
      // Update cache state
      this.cacheState[modelId] = {
        size: modelData.byteLength,
        timestamp: Date.now(),
        metadata
      };
      
      // Persist cache state
      await this._persistCacheState();
      
      // For Node.js, save model data to disk
      if (typeof window === 'undefined' && this.cacheDir) {
        try {
          const fs = require('fs');
          const path = require('path');
          const modelDir = path.join(this.cacheDir, 'models');
          if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, { recursive: true });
          }
          
          // Save model data
          const modelPath = path.join(modelDir, `${modelId.replace(/\//g, '_')}.bin`);
          fs.writeFileSync(modelPath, Buffer.from(modelData));
          
          // Save metadata
          const metadataPath = path.join(modelDir, `${modelId.replace(/\//g, '_')}.json`);
          fs.writeFileSync(
            metadataPath, 
            JSON.stringify({ ...metadata, timestamp: Date.now() }, null, 2)
          );
          
          logger.debug(`Model ${modelId} saved to disk cache`);
        } catch (err) {
          logger.warn(`Failed to save model ${modelId} to disk: ${err.message}`);
        }
      }
      
      if (this.logCacheMetrics) {
        await this._logCacheMetrics();
      }
      
      return true;
    } catch (error) {
      logger.error(`Failed to store model in custom cache: ${error.message}`);
      return false;
    }
  }
  
  async retrieveModel(modelId) {
    // Check if model exists in memory cache
    if (this.customStorage.has(modelId)) {
      try {
        // Get model from memory cache
        const cachedModel = this.customStorage.get(modelId);
        
        // Update access timestamp
        this.cacheState[modelId].lastAccessed = Date.now();
        await this._persistCacheState();
        
        logger.debug(`Retrieved model ${modelId} from memory cache`);
        return {
          data: cachedModel.data,
          metadata: cachedModel.metadata
        };
      } catch (error) {
        logger.error(`Failed to retrieve model from memory cache: ${error.message}`);
      }
    }
    
    // If not in memory, try loading from disk (Node.js only)
    if (typeof window === 'undefined' && this.cacheDir) {
      try {
        const fs = require('fs');
        const path = require('path');
        const modelPath = path.join(this.cacheDir, 'models', `${modelId.replace(/\//g, '_')}.bin`);
        const metadataPath = path.join(this.cacheDir, 'models', `${modelId.replace(/\//g, '_')}.json`);
        
        if (fs.existsSync(modelPath) && fs.existsSync(metadataPath)) {
          // Load model data and metadata
          const modelData = fs.readFileSync(modelPath);
          const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
          
          // Store in memory cache
          this.customStorage.set(modelId, {
            data: modelData,
            metadata: metadata
          });
          
          // Update access timestamp
          this.cacheState[modelId].lastAccessed = Date.now();
          await this._persistCacheState();
          
          logger.debug(`Retrieved model ${modelId} from disk cache`);
          return {
            data: modelData,
            metadata: metadata
          };
        }
      } catch (err) {
        logger.error(`Failed to retrieve model from disk cache: ${err.message}`);
      }
    }
    
    logger.debug(`Model ${modelId} not found in cache`);
    return null;
  }
  
  async evictModel(modelId) {
    let evicted = false;
    
    // Remove from memory
    if (this.customStorage.has(modelId)) {
      this.customStorage.delete(modelId);
      evicted = true;
    }
    
    // Remove from cache state
    if (this.cacheState[modelId]) {
      delete this.cacheState[modelId];
      await this._persistCacheState();
      evicted = true;
    }
    
    // Remove from disk (Node.js only)
    if (typeof window === 'undefined' && this.cacheDir) {
      try {
        const fs = require('fs');
        const path = require('path');
        const modelPath = path.join(this.cacheDir, 'models', `${modelId.replace(/\//g, '_')}.bin`);
        const metadataPath = path.join(this.cacheDir, 'models', `${modelId.replace(/\//g, '_')}.json`);
        
        if (fs.existsSync(modelPath)) {
          fs.unlinkSync(modelPath);
          evicted = true;
        }
        
        if (fs.existsSync(metadataPath)) {
          fs.unlinkSync(metadataPath);
          evicted = true;
        }
      } catch (err) {
        logger.error(`Failed to evict model from disk: ${err.message}`);
      }
    }
    
    if (evicted) {
      logger.debug(`Evicted model ${modelId} from cache`);
    }
    
    if (this.logCacheMetrics) {
      await this._logCacheMetrics();
    }
    
    return evicted;
  }
  
  async getStorageStats() {
    let totalSize = 0;
    let modelCount = 0;
    let availableSpace = Infinity;
    
    for (const modelId in this.cacheState) {
      totalSize += this.cacheState[modelId].size;
      modelCount++;
    }
    
    // Check available space
    if (typeof window === 'undefined') {
      try {
        // Node.js environment
        const { freemem } = require('os');
        availableSpace = freemem();
      } catch (err) {
        logger.warn(`Failed to get available space: ${err.message}`);
      }
    } else {
      try {
        // Browser environment
        if (navigator.storage && navigator.storage.estimate) {
          const estimate = await navigator.storage.estimate();
          availableSpace = estimate.quota - estimate.usage;
        }
      } catch (err) {
        logger.warn(`Failed to estimate storage: ${err.message}`);
      }
    }
    
    return {
      totalSize,
      modelCount,
      availableSpace
    };
  }
  
  // Private method to persist cache state
  async _persistCacheState() {
    try {
      if (typeof window !== 'undefined') {
        // Browser environment
        localStorage.setItem('model-cache-state', JSON.stringify(this.cacheState));
      } else if (this.cacheDir) {
        // Node.js environment
        const fs = require('fs');
        const path = require('path');
        const cacheStatePath = path.join(this.cacheDir, 'cache_state.json');
        fs.writeFileSync(cacheStatePath, JSON.stringify(this.cacheState, null, 2), 'utf8');
      }
    } catch (error) {
      logger.error(`Failed to persist cache state: ${error.message}`);
    }
  }
  
  // Log cache metrics to reports directory
  async _logCacheMetrics() {
    if (typeof window === 'undefined') {
      try {
        const fs = require('fs');
        const path = require('path');
        const stats = await this.getStorageStats();
        
        const metricsDir = path.join(__dirname, '../../../../reports/performance/cache');
        if (!fs.existsSync(metricsDir)) {
          fs.mkdirSync(metricsDir, { recursive: true });
        }
        
        const metrics = {
          timestamp: new Date().toISOString(),
          stats: stats,
          modelCount: Object.keys(this.cacheState).length,
          models: Object.keys(this.cacheState)
        };
        
        fs.writeFileSync(
          path.join(metricsDir, `cache_metrics_${new Date().toISOString().slice(0,10)}.json`),
          JSON.stringify(metrics, null, 2)
        );
      } catch (err) {
        logger.error(`Failed to log cache metrics: ${err.message}`);
      }
    }
  }
}
```

## Implementation Roadmap

The JavaScript SDK is following the same phased implementation strategy as the Python SDK:

1. **Phase 1: Core Architecture (June-July 2025)**
   - Implement browser hardware detection
   - Create WebGPU, WebNN, and WebAssembly backends
   - Develop core acceleration interface
   - Implement IndexedDB storage
   - Create React component integration

2. **Phase 2: Browser-Specific Enhancement (July-August 2025)**
   - Optimize for Chrome, Firefox, Edge, and Safari
   - Implement browser-specific shader optimizations
   - Develop Firefox audio compute shader optimizations
   - Implement Edge WebNN optimizations
   - Add shader precompilation for faster startup
   - Create browser capability detection system

3. **Phase 3: Advanced Feature Integration (August-September 2025)**
   - Implement ultra-low precision framework
   - Add browser-based benchmarking system
   - Implement model sharding across tabs
   - Create distributed execution framework
   - Add P2P optimization layer
   - Implement progressive loading for large models

4. **Phase 4: Finalization (September-October 2025)**
   - Complete comprehensive test suite
   - Finalize API design and ensure stability
   - Develop detailed documentation
   - Create starter templates and examples
   - Implement browser-based visualization tools
   - Ensure seamless integration with existing web frameworks

## Release Notes

### Version 0.4.0 (Current Release - March 7, 2025)

- Added WebGPU and WebNN hardware detection and acceleration
- Implemented browser-specific optimizations (Firefox for audio models)
- Added IndexedDB storage for acceleration results
- Implemented React hooks for easy integration
- Added shader precompilation for faster startup
- Enhanced P2P network optimization
- Added comprehensive browser capabilities detection

### Version 0.3.0 (Previous Release)

- Added initial WebGPU support
- Basic WebNN integration
- Initial benchmarking capabilities
- Core acceleration interface

## Further Reading

- [API Documentation](API_DOCUMENTATION.md)
- [Browser Compatibility Matrix](WEBNN_WEBGPU_COMPATIBILITY_MATRIX.md)
- [WebGPU Optimization Guide](WEBGPU_BROWSER_OPTIMIZATIONS.md)
- [React Integration Guide](WEB_PLATFORM_INTEGRATION_GUIDE.md)
- [P2P Network Optimization](P2P_NETWORK_OPTIMIZATION_GUIDE.md)
- [Ultra-Low Precision Framework](ULTRA_LOW_PRECISION_IMPLEMENTATION_GUIDE.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)

## Reporting Structure

When using the SDK in Node.js environments or when exporting reports from browser environments, the following report structure is used:

```
reports/
├── benchmarks/                  # Benchmark reports and results
│   ├── webgpu/                  # WebGPU-specific benchmarks
│   │   ├── chrome/              # Chrome browser results
│   │   ├── firefox/             # Firefox browser results
│   │   └── edge/                # Edge browser results
│   ├── webnn/                   # WebNN-specific benchmarks
│   │   └── edge/                # Edge is primary WebNN browser
│   └── wasm/                    # WebAssembly benchmarks
│       ├── text_models/         # Text model results
│       ├── vision_models/       # Vision model results
│       └── audio_models/        # Audio model results
├── compatibility/               # Browser compatibility reports
│   ├── feature_matrix/          # Feature support across browsers
│   └── performance_matrix/      # Performance comparison across browsers
├── performance/                 # Performance analysis reports
│   ├── text_models/             # Text model performance reports
│   ├── vision_models/           # Vision model performance reports
│   ├── audio_models/            # Audio model performance reports
│   └── multimodal_models/       # Multimodal model performance reports
├── accelerate/                  # IPFS acceleration reports
│   ├── ipfs_metrics/            # IPFS performance metrics
│   └── p2p_analysis/            # P2P network analysis
└── visualizations/              # Generated charts and visualizations
    ├── benchmark_charts/        # Benchmark visualization charts
    ├── browser_comparison/      # Browser comparison charts
    └── hardware_comparison/     # Hardware comparison charts
```

Report naming convention:
- `{model_name}_{benchmark_type}_{browsers}_{date}.{format}`
- Example: `bert-base-uncased_latency_chrome-firefox-edge_20250307.html`

Node.js applications will write reports to these locations by default, while browser applications will use these paths when exporting reports for download.