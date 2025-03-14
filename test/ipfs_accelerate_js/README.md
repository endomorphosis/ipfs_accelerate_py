# IPFS Accelerate JS

Hardware-accelerated machine learning for the browser with WebGPU and WebNN.

[![npm version](https://img.shields.io/npm/v/ipfs-accelerate-js.svg)](https://www.npmjs.com/package/ipfs-accelerate-js)
[![license](https://img.shields.io/github/license/ipfs/ipfs-accelerate-js.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Overview

IPFS Accelerate JS delivers high-performance AI models directly in the browser using WebGPU and WebNN for hardware acceleration. Run transformers, vision models, and audio models with GPU-accelerated inference - no server required.

### Key Features

- **Hardware Acceleration**: Uses WebGPU and WebNN for GPU-accelerated inference
- **Cross-Browser Compatibility**: Works in Chrome, Firefox, Edge, and Safari
- **Multiple Model Support**: Text (BERT), Vision (ViT), and Audio (Whisper) models
- **Cross-Model Tensor Sharing**: Share tensors between models for efficient multimodal applications
- **Advanced Memory Optimization**: 
  - Ultra-low precision quantization (1/2/3/4/8-bit precision)
  - Per-channel quantization for weights with minimal accuracy loss
  - Up to 93.75% memory reduction with 2-bit quantization
- **Operation Fusion**: 
  - Combines multiple operations into single compute shaders for performance
  - Eliminates intermediate buffers and reduces kernel launches
  - Specialized fusion patterns for transformer models (attention mechanism)
- **Browser-Specific Optimizations**: 
  - Automatic detection of Chrome, Firefox, Safari, and Edge
  - Tailored shader workgroup sizes and memory access patterns by browser
  - Specialized optimizations for different GPU architectures
- **Performance Optimizations**:
  - Tiled matrix multiplication with shared memory
  - Advanced WGSL shader compilation and caching
  - Automatic performance tuning based on input size

## Quick Start

### Installation

```bash
npm install ipfs-accelerate-js
```

### Basic Usage

```typescript
import { createVitModel, createWebGPUBackend } from 'ipfs-accelerate-js';

async function classifyImage(imageData, width, height) {
  // Create hardware backend
  const hardware = await createWebGPUBackend();
  await hardware.initialize();
  
  // Create ViT model
  const model = createVitModel(hardware, {
    modelId: 'google/vit-base-patch16-224'
  });
  await model.initialize();
  
  // Process image
  const result = await model.process({
    imageData,
    width,
    height
  });
  
  console.log(`Predicted class: ${result.classId}`);
  console.log(`Probability: ${result.probabilities[result.classId]}`);
  
  // Clean up resources
  await model.dispose();
}
```

## Models

### Vision Models

```typescript
import { createVitModel } from 'ipfs-accelerate-js';

// Create Vision Transformer (ViT) model
const vit = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224',
  useOptimizedOps: true
});

// Process image
const result = await vit.process({
  imageData,
  width: 224,
  height: 224
});
```

### Text Models

```typescript
import { createBertModel } from 'ipfs-accelerate-js';

// Create BERT model
const bert = createBertModel(hardware, {
  modelId: 'bert-base-uncased',
  useOptimizedOps: true
});

// Tokenize and process text
const tokens = await bert.tokenize("Hello world");
const result = await bert.process(tokens);
```

### Audio Models

```typescript
import { createWhisperModel } from 'ipfs-accelerate-js';

// Create Whisper model
const whisper = createWhisperModel(hardware, {
  modelId: 'openai/whisper-tiny',
  useBrowserOptimizations: true
});

// Process audio
const result = await whisper.process({
  audioData,
  sampleRate: 16000
});

console.log(`Transcription: ${result.text}`);
```

## Cross-Model Tensor Sharing

Share tensors between models for efficient multimodal applications:

```typescript
// Process image with ViT
const vitResult = await vitModel.process(imageInput);
const visionEmbedding = vitModel.getSharedTensor('vision_embedding');

// Process text with BERT 
const bertResult = await bertModel.process(tokens);
const textEmbedding = bertModel.getSharedTensor('text_embedding');

// Use both embeddings in a multimodal application
// This avoids redundant computation and reduces memory usage
```

## Hardware Backends

Choose from multiple backends for hardware acceleration:

```typescript
import { 
  createWebGPUBackend, 
  createWebNNBackend,
  getBrowserHardwareCapabilities 
} from 'ipfs-accelerate-js';

// Detect available hardware capabilities
const capabilities = await getBrowserHardwareCapabilities();

// Create appropriate backend based on capabilities
let hardware;
if (capabilities.webgpu) {
  hardware = await createWebGPUBackend();
} else if (capabilities.webnn) {
  hardware = await createWebNNBackend();
} else {
  // Fallback to CPU backend
  hardware = await createCPUBackend();
}
```

## Browser Support

| Feature | Chrome | Firefox | Edge | Safari |
|---------|--------|---------|------|--------|
| WebGPU | 113+ | 118+ | 113+ | 17+ |
| WebNN | - | - | 113+ | - |
| CPU Fallback | ✓ | ✓ | ✓ | ✓ |

## Examples

### Browser-Based Image Classification

```html
<!DOCTYPE html>
<html>
<head>
  <title>ViT Demo</title>
  <script src="https://cdn.jsdelivr.net/npm/ipfs-accelerate-js/dist/ipfs-accelerate.min.js"></script>
</head>
<body>
  <input type="file" id="image-input" accept="image/*">
  <div id="result"></div>
  
  <script>
    const { createVitModel, createWebGPUBackend } = IPFSAccelerate;
    
    document.getElementById('image-input').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      const img = await createImageBitmap(file);
      
      // Create canvas to get image data
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 224, 224);
      const imageData = ctx.getImageData(0, 0, 224, 224).data;
      
      // Convert to RGB format (remove alpha channel)
      const rgbData = new Uint8Array(224 * 224 * 3);
      for (let i = 0; i < imageData.length / 4; i++) {
        rgbData[i * 3] = imageData[i * 4];
        rgbData[i * 3 + 1] = imageData[i * 4 + 1];
        rgbData[i * 3 + 2] = imageData[i * 4 + 2];
      }
      
      // Initialize hardware and model
      const hardware = await createWebGPUBackend();
      await hardware.initialize();
      
      const model = createVitModel(hardware, {
        modelId: 'google/vit-base-patch16-224'
      });
      await model.initialize();
      
      // Process image
      const result = await model.process({
        imageData: rgbData,
        width: 224,
        height: 224
      });
      
      // Display result
      document.getElementById('result').textContent = 
        `Top class: ${result.classId} (${result.probabilities[result.classId]})`;
      
      // Clean up
      await model.dispose();
    });
  </script>
</body>
</html>
```

## Advanced Usage

### Custom Configuration

```typescript
const vitModel = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224',
  imageSize: 224,
  patchSize: 16,
  hiddenSize: 768,
  numLayers: 12,
  numHeads: 12,
  intermediateSize: 3072,
  backendPreference: ['webgpu', 'webnn', 'cpu'],
  useOptimizedOps: true
});
```

### Quantization for Memory Efficiency

```typescript
// Create model with 4-bit quantization for memory efficiency
const model = createBertModel(hardware, {
  modelId: 'bert-base-uncased',
  quantization: {
    enabled: true,
    bitsPerWeight: 4,          // 4-bit precision for weights (8x smaller than FP32)
    bitsPerActivation: 8,      // 8-bit for activations
    includeFirstLayer: false,  // Keep first layer in full precision for accuracy
    includeLastLayer: false    // Keep last layer in full precision for accuracy
  }
});

// Ultra-low precision options for extreme memory constraints
const tinyModel = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224',
  quantization: {
    enabled: true,
    bitsPerWeight: 2,          // 2-bit precision for extreme compression (16x smaller)
    symmetricQuantization: true, // Use symmetric quantization for better numeric stability
    perChannelQuantization: true // Per-channel quantization for better accuracy
  }
});
```

### Advanced Performance Optimization

You can combine quantization, operation fusion, and browser-specific optimizations for maximum performance and memory efficiency:

```typescript
// Configure comprehensive optimizations
const config = {
  // Operation fusion with quantization
  useOperationFusion: true,
  fusionOptions: {
    patterns: [
      'linear_activation',
      'elementwise_chain', 
      'attention_pattern',
      'quantized_matmul',
      'quantized_matmul_activation',
      'quantized_attention'
    ],
    useQuantizedWeights: true,
    bitsPerWeight: 3,  // Ultra-low precision (3-bit)
    useBrowserOptimizations: true
  },
  
  // Advanced quantization options
  quantization: {
    enabled: true,
    bitsPerWeight: 3,
    symmetricQuantization: true,
    perChannelQuantization: true,
    calibrationSamples: 10  // Number of samples for calibration
  },
  
  // Memory optimization
  memoryOptimization: {
    aggressiveBufferReuse: true,
    releaseIntermediateTensors: true,
    useCompression: true
  },
  
  // Browser-specific optimizations
  browserOptimizations: {
    detectBrowserAutomatically: true,
    customWorkgroupSizes: true,
    useShaderPrecompilation: true,
    adaptiveWorkloadDistribution: true
  }
};

// Create model with advanced optimizations
const model = createBertModel(hardware, {
  modelId: 'bert-base-uncased',
  ...config
});

// Process input with optimized execution
const result = await model.process(input);
```

This configuration provides:
- 90.6% memory reduction compared to FP32 (3-bit quantization)
- Specialized fusion patterns for transformer attention mechanisms
- Browser-specific optimizations for shader workgroup sizes and memory access
- Adaptive performance tuning based on input size and device capabilities

### Browser-Specific Optimizations

```typescript
import { 
  createVitModel, 
  detectBrowserType, 
  BrowserType 
} from 'ipfs-accelerate-js';

// Automatic browser detection and optimization
const model = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224',
  useBrowserOptimizations: true,  // Automatically detect and apply browser-specific optimizations
  quantization: {
    enabled: true,
    bitsPerWeight: 4,
    useBrowserOptimizations: true  // Use browser-specific quantization implementations
  }
});

// Manual browser optimization configuration
const browserType = detectBrowserType();
let config = {
  modelId: 'google/vit-base-patch16-224',
  useBrowserOptimizations: true
};

// Optionally override specific parameters
switch(browserType) {
  case BrowserType.CHROME:
    config.workgroupSize = 256;
    config.tileSize = 16;
    config.useSharedMemory = true;
    break;
  case BrowserType.FIREFOX:
    config.workgroupSize = 128;
    config.tileSize = 8;
    config.useSimpleLoops = true;
    break;
  case BrowserType.SAFARI:
    config.workgroupSize = 512;
    config.tileSize = 32;
    config.useVectorOperations = true;
    break;
  case BrowserType.EDGE:
    config.workgroupSize = 256;
    config.usePartialLoopUnrolling = true;
    config.useWebNNWhenAvailable = true;
    break;
}

const customModel = createVitModel(hardware, config);
```

You can run the browser-specific optimization benchmarks to see the impact on your system:

```bash
# Run browser-specific optimization tests and benchmarks
./run_browser_optimized_tests.sh benchmark
# Open browser to http://localhost:8080/examples/browser_specific_quantization_benchmark.html
```

### Memory Management

```typescript
// Create model in a try-finally block to ensure cleanup
let model = null;
try {
  model = createVitModel(hardware, config);
  await model.initialize();
  
  // Use the model...
  const result = await model.process(input);
  
} finally {
  // Always clean up resources
  if (model) {
    await model.dispose();
  }
}
```

## Documentation

- [API Reference](./docs/api/README.md)
- [Models](./docs/models/README.md)
- [Hardware Backends](./docs/hardware/README.md)
- [Cross-Model Tensor Sharing](./docs/CROSS_MODEL_TENSOR_SHARING.md)
- [Performance Optimization](./docs/PERFORMANCE_OPTIMIZATION.md)
- [Quantization and Operation Fusion](./docs/QUANTIZATION_OPERATION_FUSION.md) (NEW)
- [Browser-Specific Optimizations](./docs/BROWSER_OPTIMIZATIONS.md) (UPDATED)
- [Benchmark Visualization Guide](./docs/VISUALIZATION_GUIDE.md) (NEW)
- [Ultra-Low Precision](./docs/ULTRA_LOW_PRECISION.md)
- [Quantization Guide](./docs/QUANTIZATION_GUIDE.md)
- [Operation Fusion](./docs/OPERATION_FUSION.md)
- [Memory Optimization](./docs/MEMORY_OPTIMIZATION.md)
- [Examples](./examples/README.md)

## Testing and Performance Analysis

We provide comprehensive tests and performance analysis tools to validate the functionality and performance of the library:

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Measure performance across different configurations
- **Browser Compatibility Tests**: Test across different browsers
- **Visual Performance Analysis**: Interactive visualization of performance metrics

### Notable Test Files

- [WebGPU Matrix Operations Test](./test/webgpu_matrix_operations_test.ts)
- [Fusion and Quantization Integration Test](./test/fusion_quantization_test.ts) (NEW)
- [Browser-Specific Shaders Test](./test/browser_specific_shaders_test.ts) (NEW)
- [Tensor Sharing Integration Test](./test/tensor_sharing_integration.test.ts)

### Performance Visualization Tools

- [Benchmark Visualization](./examples/benchmark_visualization.html)
  - Interactive comparison of optimization techniques
  - Performance analysis across browsers
  - Memory efficiency visualization
  - Accuracy vs. memory tradeoff analysis
  
- [Browser-Specific Quantization Benchmark](./examples/browser_specific_quantization_benchmark.html) (NEW)
  - Compare browser-specific optimization performance
  - Visualize memory reduction across bit-widths
  - Analyze operation fusion performance
  - Interactive comparison of different activation functions
  - Browser-specific parameter visualization

```bash
# Run core tests
npm test

# Run browser-specific tests
npm test -- test/browser_specific_shaders_test.ts
npm test -- test/browser_specific_fusion_quantization_test.ts
npm test -- test/fusion_quantization_test.ts

# Run benchmarks and visualize results
./run_browser_optimized_tests.sh benchmark
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use IPFS Accelerate JS in your research, please cite:

```bibtex
@software{ipfs_accelerate_js,
  author = {IPFS Accelerate Team},
  title = {IPFS Accelerate JS: Hardware-accelerated machine learning for the browser},
  url = {https://github.com/ipfs/ipfs-accelerate-js},
  year = {2025},
}
```