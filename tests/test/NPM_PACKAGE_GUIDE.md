# IPFS Accelerate NPM Package Guide

**Date:** March 15, 2025  
**Version:** 1.0.0  
**Status:** Ready for Publication

This guide provides comprehensive information about the IPFS Accelerate NPM package, which enables hardware-accelerated AI in web browsers using WebGPU and WebNN.

## Table of Contents

1. [Installation](#installation)
2. [Key Features](#key-features)
3. [Basic Usage](#basic-usage)
4. [Hardware Abstraction Layer](#hardware-abstraction-layer)
5. [Model Implementations](#model-implementations)
6. [Cross-Model Tensor Sharing](#cross-model-tensor-sharing)
7. [Browser Compatibility](#browser-compatibility)
8. [Bundle Sizes](#bundle-sizes)
9. [Examples](#examples)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)
12. [Support and Community](#support-and-community)

## Installation

### NPM

```bash
npm install ipfs-accelerate
```

### Yarn

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

## Key Features

- **Hardware Abstraction Layer**: Unified interface for WebGPU, WebNN, and CPU backends
- **Browser-Specific Optimizations**: Enhanced performance in Chrome, Firefox, Edge, and Safari
- **Cross-Model Tensor Sharing**: Share tensors between models for improved memory efficiency
- **Hardware-Abstracted Models**: Automatic backend selection for BERT, ViT, Whisper, and CLIP
- **Operation Fusion**: Improved performance through operation fusion
- **TypeScript Support**: Comprehensive TypeScript definitions

## Basic Usage

### Tensor Operations

```typescript
import { Tensor, createHardwareAbstractionLayer } from 'ipfs-accelerate';

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

### Using a Pre-trained Model

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

## Hardware Abstraction Layer

The Hardware Abstraction Layer (HAL) provides a unified interface for hardware acceleration across different backends:

```typescript
import { createHardwareAbstractionLayer } from 'ipfs-accelerate';

// Create HAL with options
const hal = await createHardwareAbstractionLayer({
  // Custom backend order
  backendOrder: ['webgpu', 'webnn', 'cpu'],
  
  // Model-specific preferences
  modelPreferences: {
    'vision': 'webgpu',
    'text': 'webnn',
    'audio': 'webgpu'
  },
  
  // Backend-specific options
  backendOptions: {
    'webgpu': { 
      powerPreference: 'high-performance',
      shaderCompilation: { precompile: true }
    },
    'webnn': { 
      deviceType: 'gpu' 
    },
    'cpu': { 
      useWebWorkers: true,
      maxWorkers: 4
    }
  },
  
  // Enable tensor sharing between models
  enableTensorSharing: true,
  
  // Enable operation fusion for better performance
  enableOperationFusion: true
});
```

### Available Operations

The HAL supports a wide range of operations:

- **Basic Arithmetic**: add, subtract, multiply, divide
- **Matrix Operations**: matmul, transpose, reshape
- **Neural Network Operations**: relu, sigmoid, tanh, gelu, softmax
- **Layer Operations**: layerNorm, ffnn (feed-forward neural network)
- **Reduction Operations**: reduceSum, reduceMean
- **Tensor Manipulation**: slice, concat, gather, repeat

See the [API Reference](API_REFERENCE.md) for a complete list of supported operations.

## Model Implementations

The package includes hardware-abstracted implementations of popular AI models:

### BERT (Text Model)

```typescript
import { createHardwareAbstractedBERT } from 'ipfs-accelerate';

// Create BERT model
const bert = await createHardwareAbstractedBERT({
  modelName: 'bert-base-uncased',
  maxLength: 512,
  quantized: true,
  enableTensorSharing: true
});

// Get text embeddings
const embeddings = await bert.getEmbeddings('Input text');

// Run sentiment analysis
const sentiment = await bert.analyzeSentiment('Input text');

// Run question answering
const answer = await bert.answerQuestion({
  question: 'What is the capital of France?',
  context: 'Paris is the capital and most populous city of France.'
});
```

### ViT (Vision Model)

```typescript
import { createHardwareAbstractedViT } from 'ipfs-accelerate';

// Create ViT model
const vit = await createHardwareAbstractedViT({
  modelName: 'vit-base-patch16-224',
  imageSize: 224,
  quantized: true
});

// Classify an image
const image = await loadImageAsArray('image.jpg');
const classification = await vit.classify(image);

// Get image embeddings
const embeddings = await vit.getEmbeddings(image);
```

### Whisper (Audio Model)

```typescript
import { createHardwareAbstractedWhisper } from 'ipfs-accelerate';

// Create Whisper model
const whisper = await createHardwareAbstractedWhisper({
  modelName: 'whisper-tiny',
  language: 'en',
  quantized: true
});

// Transcribe audio
const audio = await loadAudioAsArray('audio.mp3');
const transcript = await whisper.transcribe(audio);

// Translate audio
const translation = await whisper.translate(audio, { targetLanguage: 'fr' });
```

### CLIP (Multimodal Model)

```typescript
import { createHardwareAbstractedCLIP } from 'ipfs-accelerate';

// Create CLIP model
const clip = await createHardwareAbstractedCLIP({
  modelName: 'clip-vit-base-patch32',
  quantized: true
});

// Get image-text similarity
const image = await loadImageAsArray('image.jpg');
const similarity = await clip.getSimilarity(image, 'A dog running in a field');

// Search images with text
const bestMatch = await clip.findBestMatch('A dog running in a field', [
  await loadImageAsArray('image1.jpg'),
  await loadImageAsArray('image2.jpg'),
  await loadImageAsArray('image3.jpg')
]);
```

## Cross-Model Tensor Sharing

The package includes a powerful cross-model tensor sharing system for improved memory efficiency:

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
console.log('Sharing opportunities:', stats.sharingOpportunities);
```

## Browser Compatibility

The package is compatible with all major browsers that support WebGPU and/or WebNN:

| Browser | WebGPU | WebNN | Best For |
|---------|--------|-------|----------|
| Chrome  | ✓      | ⚠️ (Limited) | Vision models, general use |
| Edge    | ✓      | ✓      | Text models, WebNN acceleration |
| Firefox | ✓      | ✗      | Audio models (excellent compute shaders) |
| Safari  | ⚠️ (Limited) | ✗      | Basic acceleration with fallbacks |

The Hardware Abstraction Layer automatically selects the best backend based on the browser capabilities and model type.

## Bundle Sizes

The package is available in multiple bundle formats to optimize size:

| Bundle | Size (min) | Size (min+gzip) | Description |
|--------|------------|-----------------|-------------|
| Full Bundle | 360 KB | 120 KB | All features and models |
| Core Only | 85 KB | 28 KB | Core functionality without models |
| WebGPU Only | 120 KB | 40 KB | WebGPU backend only |
| WebNN Only | 110 KB | 38 KB | WebNN backend only |
| BERT | 65 KB | 22 KB | BERT model only |
| ViT | 60 KB | 20 KB | ViT model only |
| Whisper | 70 KB | 24 KB | Whisper model only |
| CLIP | 75 KB | 25 KB | CLIP model only |

You can import only the specific components you need to minimize bundle size:

```typescript
// Import only the BERT model
import { createHardwareAbstractedBERT } from 'ipfs-accelerate/dist/models/bert';

// Import only the WebGPU backend
import { createWebGPUBackend } from 'ipfs-accelerate/dist/backends/webgpu';
```

## Examples

The package includes a variety of examples to help you get started:

| Example | Description |
|---------|-------------|
| [Basic Tensor Operations](examples/basic-tensor-operations.html) | Simple tensor operations with HAL |
| [BERT Text Classification](examples/bert-text-classification.html) | Text classification with BERT |
| [ViT Image Classification](examples/vit-image-classification.html) | Image classification with ViT |
| [Whisper Transcription](examples/whisper-transcription.html) | Audio transcription with Whisper |
| [CLIP Image Search](examples/clip-image-search.html) | Image search with CLIP |
| [Cross-Model Tensor Sharing](examples/cross-model-tensor-sharing.html) | Memory optimization with tensor sharing |
| [Browser Optimization Comparison](examples/browser-optimization-comparison.html) | Browser-specific optimizations |

All examples are available in the [examples](examples/) directory and on the [documentation website](https://ipfs-accelerate.github.io/examples/).

## API Reference

For a complete API reference, see the [API Reference](API_REFERENCE.md) documentation or the [online API reference](https://ipfs-accelerate.github.io/api/).

## Troubleshooting

### Common Issues

#### WebGPU Not Available

If WebGPU is not available in your browser:

```typescript
// Check if WebGPU is available
if (!navigator.gpu) {
  console.warn('WebGPU is not available in this browser');
  // Use CPU backend instead
  const hal = await createHardwareAbstractionLayer({
    backendOrder: ['cpu']
  });
}
```

#### Memory Management

To avoid memory leaks, always dispose of resources when you're done:

```typescript
// Create HAL
const hal = await createHardwareAbstractionLayer();

try {
  // Use HAL for operations
  // ...
} finally {
  // Always dispose of resources
  hal.dispose();
}
```

#### Performance Optimization

For best performance:

1. Use operation fusion when possible
2. Use the appropriate backend for your model type
3. Enable tensor sharing for multiple models
4. Consider using quantized models for faster inference
5. Use browser-specific optimizations

For more troubleshooting tips, see the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md).

## Support and Community

- **GitHub Issues**: [Report issues](https://github.com/ipfs-accelerate/ipfs-accelerate-js/issues)
- **Discussion Forum**: [Join discussions](https://github.com/ipfs-accelerate/ipfs-accelerate-js/discussions)
- **Documentation**: [Read documentation](https://ipfs-accelerate.github.io)
- **Examples**: [View examples](https://ipfs-accelerate.github.io/examples/)

## License

This package is licensed under the [MIT License](LICENSE).