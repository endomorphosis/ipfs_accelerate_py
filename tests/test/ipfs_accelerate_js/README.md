# IPFS Accelerate JavaScript SDK

[![npm version](https://img.shields.io/npm/v/ipfs-accelerate.svg)](https://www.npmjs.com/package/ipfs-accelerate)
[![license](https://img.shields.io/github/license/ipfs-accelerate/ipfs-accelerate-js)](https://github.com/ipfs-accelerate/ipfs-accelerate-js/blob/main/LICENSE)
[![bundle size](https://img.shields.io/bundlephobia/minzip/ipfs-accelerate)](https://bundlephobia.com/package/ipfs-accelerate)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.9%2B-blue)](https://www.typescriptlang.org/)

Hardware-accelerated machine learning in browsers with WebGPU and WebNN.

## Features

- **Hardware Abstraction Layer (HAL)**: Unified interface for WebGPU, WebNN, and CPU backends
- **Browser-Specific Optimizations**: Enhanced performance in Chrome, Firefox, Edge, and Safari
- **Cross-Model Tensor Sharing**: Share tensors between models for improved memory efficiency (25-40% memory reduction)
- **Error Recovery System**: Intelligent error handling with multiple recovery strategies:
  - Automatic backend switching based on performance history
  - Operation fallback to alternative implementations
  - Browser-specific recovery techniques
  - Parameter adjustment for compatibility issues
- **Performance Tracking & Analysis**: Automatic operation performance recording and statistical analysis
- **API Backends Integration**:
  - **VLLM Unified**: Advanced VLLM integration with batch processing, streaming, and model management
  - **OpenAI**: Full support for GPT models
  - **Claude**: Integration with Anthropic Claude models
  - **Hugging Face**: TGI and TEI integrations
- **Hardware-Abstracted Models**: 
  - **BERT**: Text understanding and embeddings
  - **ViT**: Image classification and embeddings
  - **Whisper**: Audio transcription and processing
  - **CLIP**: Multimodal vision-text understanding
- **Performance Optimizations**:
  - Up to 6.5x speedup vs. CPU for vision models
  - Up to 5.8x speedup vs. CPU for text models
  - Up to 3.0x speedup vs. CPU for audio models
  - Operation fusion for improved performance
  - Ultra-low precision (1-8 bit) quantization
  - Browser-specific shader optimizations

## Installation

```bash
npm install ipfs-accelerate
```

or

```bash
yarn add ipfs-accelerate
```

### CDN

```html
<!-- UMD build (all features) -->
<script src="https://unpkg.com/ipfs-accelerate@1.0.0/dist/ipfs-accelerate.min.js"></script>

<!-- ESM build (use with import) -->
<script type="module">
  import { createHardwareAbstractionLayer } from 'https://cdn.jsdelivr.net/npm/ipfs-accelerate@1.0.0/dist/ipfs-accelerate.esm.min.js';
</script>
```

## Basic Usage

### Hardware Abstraction Layer

```typescript
import { createHardwareAbstractionLayer } from 'ipfs-accelerate';

async function runMatrixMultiplication() {
  // Create hardware abstraction layer
  const hal = await createHardwareAbstractionLayer();
  
  try {
    // Create tensors
    const a = await hal.createTensor({
      dimensions: [2, 3],
      data: new Float32Array([1, 2, 3, 4, 5, 6]),
      dtype: 'float32'
    });
    
    const b = await hal.createTensor({
      dimensions: [3, 2],
      data: new Float32Array([7, 8, 9, 10, 11, 12]),
      dtype: 'float32'
    });
    
    // Execute matrix multiplication
    const result = await hal.matmul(a, b, { useOptimization: true });
    
    console.log('Result shape:', result.dimensions);
    console.log('Result data:', await result.getData());
  } finally {
    // Always clean up resources
    hal.dispose();
  }
}
```

### Using Hardware-Abstracted Models

```typescript
import { createHardwareAbstractedBERT } from 'ipfs-accelerate';

async function runBertModel() {
  // Create BERT model with hardware abstraction
  const bert = await createHardwareAbstractedBERT({
    modelName: 'bert-base-uncased',
    quantized: true  // Use quantized model for better performance
  });
  
  try {
    // Get text embeddings
    const embeddings = await bert.getEmbeddings('Hello, world!');
    console.log('Embeddings shape:', embeddings.dimensions);
    
    // Run sentiment analysis
    const sentiment = await bert.analyzeSentiment('I love this product!');
    console.log('Sentiment:', sentiment);
  } finally {
    // Clean up resources
    bert.dispose();
  }
}
```

### Cross-Model Tensor Sharing

```typescript
import { 
  createHardwareAbstractedViT, 
  createHardwareAbstractedBERT,
  TensorSharingManager 
} from 'ipfs-accelerate';

// Create tensor sharing manager
const sharingManager = new TensorSharingManager();

// Create models with tensor sharing
const vit = await createHardwareAbstractedViT({
  modelName: 'vit-base-patch16-224',
  tensorSharingManager: sharingManager
});

const bert = await createHardwareAbstractedBERT({
  modelName: 'bert-base-uncased',
  tensorSharingManager: sharingManager
});

// Process image and text
const image = await loadImageAsArray('image.jpg');
const imageEmbeddings = await vit.getEmbeddings(image);

const text = 'A description of the image';
const textEmbeddings = await bert.getEmbeddings(text);

// Get memory usage statistics
const stats = sharingManager.getStats();
console.log('Memory saved:', stats.memorySaved, 'bytes');
```

## Browser Compatibility

| Browser | WebGPU | WebNN | Best For |
|---------|--------|-------|----------|
| Chrome  | ✓      | ⚠️ (Limited) | Vision models, general use |
| Edge    | ✓      | ✓      | Text models, WebNN acceleration |
| Firefox | ✓      | ✗      | Audio models (excellent compute shaders) |
| Safari  | ⚠️ (Limited) | ✗      | Basic acceleration with fallbacks |

The Hardware Abstraction Layer automatically selects the best backend based on the browser capabilities and model type.

## Error Recovery System

The Error Recovery System provides intelligent error handling for WebGPU/WebNN operations, making your machine learning applications more robust and resilient.

```typescript
import { createHardwareAbstractionLayer, createErrorRecoveryManager } from 'ipfs-accelerate';

// Create hardware abstraction layer
const hal = await createHardwareAbstractionLayer();

// Create error recovery manager
const errorRecoveryManager = createErrorRecoveryManager(hal.performanceTracker);

// Protect a critical operation with error recovery
const protectedMatmul = errorRecoveryManager.protect(
  async (a, b) => await hal.matmul(a, b),
  {
    operationName: 'matmul',
    backendType: hal.getBackendType() as any,
    availableBackends: hal.backends,
    activeBackend: hal.getActiveBackend()!,
    performanceTracker: hal.performanceTracker,
    setActiveBackend: (backend) => hal.setActiveBackend(backend),
    browserType: 'chrome',
    useBrowserOptimizations: true
  }
);

// Use the protected operation
try {
  const result = await protectedMatmul(tensorA, tensorB);
  console.log('Operation succeeded!');
} catch (error) {
  console.error('All recovery attempts failed:', error);
}

// Get recovery statistics
const stats = errorRecoveryManager.getStrategySuccessRates();
console.log('Recovery success rates:', stats);
```

## Documentation

For detailed documentation, see the following resources:

- [Hardware Abstraction Layer Guide](https://ipfs-accelerate.github.io/docs/hardware-abstraction-layer)
- [Error Recovery Guide](https://ipfs-accelerate.github.io/docs/error-recovery)
- [Performance Tracking Guide](https://ipfs-accelerate.github.io/docs/performance-tracking)
- [Cross-Model Tensor Sharing Guide](https://ipfs-accelerate.github.io/docs/cross-model-tensor-sharing)
- [Browser Optimization Guide](https://ipfs-accelerate.github.io/docs/browser-optimization)
- [API Reference](https://ipfs-accelerate.github.io/api)
- [Examples](https://ipfs-accelerate.github.io/examples)

## Bundle Sizes

The package is available in multiple bundle formats to optimize size:

| Bundle | Size (min+gzip) | Import Path |
|--------|----------------|------------|
| Full Bundle | 120 KB | `import * from 'ipfs-accelerate'` |
| Core Only | 28 KB | `import * from 'ipfs-accelerate/dist/core'` |
| WebGPU Only | 40 KB | `import * from 'ipfs-accelerate/dist/backends/webgpu'` |
| WebNN Only | 38 KB | `import * from 'ipfs-accelerate/dist/backends/webnn'` |
| HAL Only | 58 KB | `import * from 'ipfs-accelerate/dist/hal'` |
| Error Recovery | 18 KB | `import * from 'ipfs-accelerate/dist/error-recovery'` |
| Performance Tracking | 16 KB | `import * from 'ipfs-accelerate/dist/performance'` |
| BERT | 22 KB | `import * from 'ipfs-accelerate/dist/models/bert'` |
| ViT | 20 KB | `import * from 'ipfs-accelerate/dist/models/vit'` |
| Whisper | 24 KB | `import * from 'ipfs-accelerate/dist/models/whisper'` |
| CLIP | 25 KB | `import * from 'ipfs-accelerate/dist/models/clip'` |

## Examples

The package includes a variety of examples to help you get started:

- [Basic Tensor Operations](https://ipfs-accelerate.github.io/examples/basic-tensor-operations)
- [BERT Text Classification](https://ipfs-accelerate.github.io/examples/bert-text-classification)
- [ViT Image Classification](https://ipfs-accelerate.github.io/examples/vit-image-classification)
- [Whisper Transcription](https://ipfs-accelerate.github.io/examples/whisper-transcription)
- [CLIP Image Search](https://ipfs-accelerate.github.io/examples/clip-image-search)
- [Cross-Model Tensor Sharing](https://ipfs-accelerate.github.io/examples/cross-model-tensor-sharing)
- [Error Recovery System](https://ipfs-accelerate.github.io/examples/error-recovery-system)
- [Performance Tracking](https://ipfs-accelerate.github.io/examples/performance-tracking)
- [Browser Optimization Comparison](https://ipfs-accelerate.github.io/examples/browser-optimization-comparison)

## Development

```bash
# Clone the repository
git clone https://github.com/ipfs-accelerate/ipfs-accelerate-js.git
cd ipfs-accelerate-js

# Install dependencies
npm install

# Build the library
npm run build

# Run tests
npm test

# Start the development server with examples
npm run start:examples
```

## License

This package is licensed under the [MIT License](LICENSE).