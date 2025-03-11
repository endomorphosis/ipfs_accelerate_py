# IPFS Accelerate JavaScript SDK Documentation

> **Version:** 0.1.0  
> **Release Date:** March 2025  
> **Status:** Beta  

## Overview

The IPFS Accelerate JavaScript SDK provides hardware-accelerated machine learning capabilities for web browsers and Node.js environments. It leverages modern web APIs such as WebGPU and WebNN to provide optimal performance across a variety of hardware platforms and browser environments.

This SDK is the TypeScript/JavaScript counterpart to the Python-based IPFS Accelerate framework, sharing the same architectural principles while being fully optimized for browser and Node.js environments.

## Key Features

- **Hardware Acceleration**: Utilizes WebGPU and WebNN for optimal performance
- **Browser-Specific Optimizations**: Tailored optimizations for Chrome, Firefox, Edge, and Safari
- **Ultra-Low Precision Support**: 2-bit to 16-bit quantization for efficient inference
- **Cross-Browser Compatibility**: Works across all major browsers
- **React Integration**: Dedicated hooks for React applications
- **Resource Pooling**: Efficient management of browser resources with automatic recovery
- **Cross-Model Tensor Sharing**: Share tensors between models for improved memory efficiency
- **IPFS Integration**: P2P model distribution and caching

## Installation

```bash
npm install ipfs-accelerate
```

## Quick Start

### Basic Usage

```javascript
import { createAccelerator } from 'ipfs-accelerate';

async function runInference() {
  // Create accelerator with automatic hardware detection
  const accelerator = await createAccelerator({
    autoDetectHardware: true
  });
  
  // Run inference
  const result = await accelerator.accelerate({
    modelId: 'bert-base-uncased',
    modelType: 'text',
    input: 'This is a sample text for embedding.'
  });
  
  console.log(result);
}

runInference();
```

### React Integration

```jsx
import React, { useState } from 'react';
import { useAccelerator } from 'ipfs-accelerate/react';

function TextEmbeddingComponent() {
  const { model, loading, error } = useAccelerator({
    modelId: 'bert-base-uncased',
    modelType: 'text'
  });
  
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (model && input) {
      const embedding = await model.embed(input);
      setResult(embedding);
    }
  };
  
  return (
    <div>
      {loading && <p>Loading model...</p>}
      {error && <p>Error: {error.message}</p>}
      {model && (
        <form onSubmit={handleSubmit}>
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder="Enter text to embed"
          />
          <button type="submit">Generate Embedding</button>
        </form>
      )}
      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
```

## Architecture

The SDK is organized into the following key components:

### Hardware Abstraction

The hardware abstraction layer provides a unified interface for interacting with different hardware backends:

```typescript
import { createHardwareContext } from 'ipfs-accelerate/hardware';

// Create hardware context with automatic detection
const hardwareContext = await createHardwareContext();

// Or specify preferred backends
const customContext = await createHardwareContext({
  preferredBackends: ['webgpu', 'webnn', 'wasm', 'cpu'],
  devicePreferences: {
    'webgpu': { preferLowPower: false },
    'webnn': { allowFallback: true }
  }
});
```

### Model Management

The model management system handles loading, caching, and optimizing models:

```typescript
import { ModelManager } from 'ipfs-accelerate/model';

// Create model manager with default options
const modelManager = new ModelManager();

// Load a model
const bertModel = await modelManager.loadModel('bert-base-uncased', {
  quantization: { bits: 4, scheme: 'symmetric' },
  caching: { enableLocalCache: true, cacheTTL: 86400 }
});

// Run inference
const result = await bertModel.embed('Sample text for embedding');
```

### Resource Pool

The resource pool system manages browser resources efficiently:

```typescript
import { ResourcePool } from 'ipfs-accelerate/browser';

// Create resource pool with fault tolerance
const resourcePool = new ResourcePool({
  maxConnections: 4,
  enableFaultTolerance: true,
  recoveryStrategy: 'progressive',
  browserPreferences: {
    audio: 'firefox',
    vision: 'chrome',
    text: 'edge'
  }
});

// Get model from pool
const model = await resourcePool.getModel('bert-base-uncased', {
  modelType: 'text',
  hardwarePreferences: {
    priorityList: ['webgpu', 'cpu']
  }
});
```

### Browser-Specific Optimizations

The SDK includes browser-specific optimizations for different model types:

| Browser | Best For | Optimizations |
|---------|----------|--------------|
| Firefox | Audio models | Optimized audio compute shaders, audio FFT operations |
| Edge | WebNN models | Superior WebNN graph optimization, INT8 acceleration |
| Chrome | Vision models | Efficient texture operations, vision model acceleration |
| Safari | Memory efficiency | Optimized memory usage, efficient caching |

These optimizations are automatically applied based on the detected browser.

## Advanced Features

### Cross-Model Tensor Sharing

The cross-model tensor sharing system enables efficient sharing of tensors between multiple models:

```typescript
import { createTensorSharingContext } from 'ipfs-accelerate/tensor';

// Create tensor sharing context
const sharingContext = createTensorSharingContext();

// Load models with sharing enabled
const bertModel = await modelManager.loadModel('bert-base-uncased', {
  sharingContext,
  sharingConfig: { shareEmbeddings: true }
});

const t5Model = await modelManager.loadModel('t5-small', {
  sharingContext,
  sharingConfig: { shareEmbeddings: true }
});

// Tensors will be automatically shared between compatible models
```

### Ultra-Low Precision Quantization

The SDK supports ultra-low precision quantization for efficient inference:

```typescript
import { QuantizationEngine } from 'ipfs-accelerate/quantization';

// Create quantization engine
const quantEngine = new QuantizationEngine();

// Quantize model to 4-bit precision
const quantizedModel = await quantEngine.quantize(model, {
  bits: 4,
  scheme: 'symmetric',
  granularity: 'per-channel',
  calibrationData: calibrationSamples
});
```

### WebGPU Shader Customization

For advanced users, the SDK allows customization of WebGPU shaders:

```typescript
import { ShaderRegistry } from 'ipfs-accelerate/worker/webgpu';

// Register custom shader for specific operation
ShaderRegistry.registerShader('matmul', customShaderCode, {
  workgroupSize: [8, 8, 1],
  browserTarget: 'firefox'
});
```

## Browser Compatibility

| Browser | WebGPU | WebNN | Tensor Sharing | Ultra-Low Precision |
|---------|--------|-------|----------------|---------------------|
| Chrome 120+ | ✅ | ✅ | ✅ | ✅ |
| Firefox 123+ | ✅ | ❌ | ✅ | ✅ |
| Edge 120+ | ✅ | ✅ | ✅ | ✅ |
| Safari 17.4+ | ✅ | ❌ | ✅ | ✅ |

## Supported Models

The SDK includes optimized implementations for the following model types:

### Text Models
- BERT (and variants)
- T5 (and variants)
- LLAMA (and variants)
- GPT-2
- Text embeddings

### Vision Models
- ViT (Vision Transformer)
- CLIP (and variants)
- DETR (and variants)
- Image classification models

### Audio Models
- Whisper (Speech-to-text)
- CLAP (Contrastive Language-Audio Pretraining)
- Audio classification models

## Performance Considerations

For optimal performance:

1. **Browser Selection**: Choose the appropriate browser for your model type (Firefox for audio, Chrome/Edge for vision)
2. **Quantization**: Use 4-bit or 8-bit quantization for significant performance improvements
3. **Tensor Sharing**: Enable tensor sharing when using multiple related models
4. **Resource Pooling**: Use the resource pool for managing multiple models

## Development and Testing

For SDK development and testing:

```bash
# Clone the repository
git clone https://github.com/your-org/ipfs-accelerate-js.git
cd ipfs-accelerate-js

# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test

# Generate documentation
npm run docs
```

## Contributing

Contributions to the IPFS Accelerate JavaScript SDK are welcome. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) guide for details on how to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The WebGPU and WebNN Working Groups
- Contributors to the IPFS Accelerate Python Framework
- The open-source ML community