# IPFS Accelerate JavaScript SDK Documentation

## Overview

The IPFS Accelerate JavaScript SDK provides hardware-accelerated machine learning for web browsers and Node.js environments. This SDK leverages WebGPU and WebNN for optimal performance across various hardware platforms.

## Developer Notes on Code Generation

This SDK is generated from Python code using a specialized Python-to-TypeScript converter. Rather than manually fixing issues in the generated TypeScript files, developers should focus on improving the generator itself:

1. **Update the Pattern Mapping**: Enhance the pattern mapping in `setup_ipfs_accelerate_js_py_converter.py` to produce correct TypeScript code.

2. **Class Templates**: Use specialized templates for WebGPU and WebNN classes in the `CLASS_CONVERSIONS` dictionary of the generator.

3. **Import Path Management**: Update the import path resolution logic in the generator to ensure correct module references.

4. **Type Inference**: Enhance type inference from Python type hints to TypeScript type annotations.

This approach creates a sustainable workflow where improvements to the generator benefit all future conversions, eliminating the need to repeatedly fix the same issues in generated code.

## Features

- **WebGPU and WebNN Support**: Utilize browser GPU and neural network accelerators.
- **Cross-Browser Compatibility**: Works across Chrome, Firefox, Safari, and Edge.
- **Multiple Model Support**: Run BERT, ViT, CLIP, Whisper, LLAMA, and more.
- **Ultra-Low Precision**: Support for 2-bit to 16-bit quantization.
- **Tensor Sharing**: Efficient memory usage by sharing tensors between models.
- **Hardware Detection**: Automatic selection of optimal hardware backends.
- **Resource Pooling**: Concurrent execution of multiple models across browser backends.
- **Fault Tolerance**: Recover from browser crashes or disconnections automatically.
- **React Integration**: Simple hooks for React applications.

## Installation

```bash
npm install ipfs-accelerate-js
```

## Quick Start

```javascript
import { createAccelerator } from 'ipfs-accelerate-js';

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

## Browser-Specific Optimizations

The SDK automatically optimizes model execution based on browser capabilities:

| Browser | Best For | Features |
|---------|----------|----------|
| Firefox | Audio models | 20-25% better performance for Whisper, CLAP |
| Edge | WebNN models | Superior WebNN implementation |
| Chrome | Vision models | Solid all-around WebGPU support |
| Safari | Text models | Good balance of performance and battery life |

## Hardware Abstraction Layer

The SDK uses a hardware abstraction layer to automatically select the best hardware backend for each model:

```javascript
import { HardwareAbstraction } from 'ipfs-accelerate-js';

// Create hardware abstraction
const hardware = new HardwareAbstraction({
  preferredBackends: ['webgpu', 'webnn', 'wasm', 'cpu']
});

// Initialize hardware detection
await hardware.initialize();

// Get optimal backend for a model type
const backend = hardware.getOptimalBackendForModel('vision');
console.log(`Optimal backend for vision models: ${backend}`);
```

## Web Resource Pool

The Web Resource Pool enables concurrent execution of multiple models across different browser backends:

```javascript
import { ResourcePoolManager } from 'ipfs-accelerate-js';

// Create resource pool
const pool = new ResourcePoolManager({
  maxConnections: 4,
  browserPreferences: {
    'audio': 'firefox',
    'vision': 'chrome',
    'text': 'edge'
  },
  enableFaultTolerance: true
});

// Initialize pool
await pool.initialize();

// Get model from pool
const model = await pool.getModel({
  modelId: 'bert-base-uncased',
  type: 'text'
});

// Run inference
const result = await model.process('This is a test');
```

## Cross-Model Tensor Sharing

Share tensors between models to improve memory efficiency:

```javascript
import { TensorSharingManager } from 'ipfs-accelerate-js';

// Create tensor sharing manager
const tensorManager = new TensorSharingManager();

// Register models
tensorManager.registerModel('text-embedding', textEmbeddingModel);
tensorManager.registerModel('text-classification', classificationModel);

// Run inference with shared tensors
const embeddings = await tensorManager.runWithSharing(
  'text-embedding',
  'This is a shared text input'
);

const classification = await tensorManager.runWithSharing(
  'text-classification',
  'This is a shared text input'
);
```

## React Integration

The SDK provides React hooks for easy integration:

```jsx
import { useModel } from 'ipfs-accelerate-js/react';

function TextClassification() {
  const { model, loading, error } = useModel({
    modelId: 'bert-base-uncased',
    modelType: 'text'
  });
  
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (model && input) {
      const classification = await model.classify(input);
      setResult(classification);
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
            placeholder="Enter text to classify"
          />
          <button type="submit">Classify</button>
        </form>
      )}
      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
```

## Quantization

Reduce model size and improve performance with quantization:

```javascript
import { QuantizationEngine } from 'ipfs-accelerate-js';

// Create quantization engine
const quantEngine = new QuantizationEngine();

// Quantize model to 8-bit precision
const quantizedModel = await quantEngine.quantize({
  modelId: 'bert-base-uncased',
  bits: 8,
  calibrationData: calibrationSamples,
  mixedPrecision: true
});

// Run inference with quantized model
const result = await quantizedModel.process('This is a test');
```

## Ultra-Low Precision

For even greater performance, the SDK supports ultra-low precision:

```javascript
import { UltraLowPrecisionEngine } from 'ipfs-accelerate-js';

// Create ULP engine
const ulpEngine = new UltraLowPrecisionEngine();

// Quantize to 4-bit precision
const ulpModel = await ulpEngine.quantize({
  modelId: 'bert-base-uncased',
  bits: 4,
  calibrationData: calibrationSamples
});

// Run inference
const result = await ulpModel.process('This is a test');
```

## Advanced Configurations

Configure the SDK for specific use cases:

```javascript
const accelerator = await createAccelerator({
  // Hardware preferences
  preferredBackend: 'webgpu',
  fallbackOrder: ['webgpu', 'webnn', 'wasm', 'cpu'],
  
  // Performance settings
  enableFaultTolerance: true,
  recoveryStrategy: 'progressive',
  stateSync: true,
  
  // Memory optimization
  enableTensorSharing: true,
  
  // Quantization defaults
  defaultQuantization: {
    bits: 8,
    scheme: 'symmetric',
    mixedPrecision: true
  },
  
  // Logging and monitoring
  logging: true,
  metrics: true
});
```

## Browser Compatibility

| Feature | Chrome | Firefox | Edge | Safari |
|---------|--------|---------|------|--------|
| WebGPU | 113+ | 118+ | 113+ | 17+ |
| WebNN | Experimental | No | Experimental | No |
| WebAssembly SIMD | Yes | Yes | Yes | Yes |
| Shared Array Buffer | Yes | Yes | Yes | Yes |

## Model Compatibility

| Model Type | Compatible Models |
|------------|-------------------|
| Text | BERT, T5, LLAMA, GPT-2, RoBERTa |
| Vision | ViT, CLIP, DETR, MobileViT |
| Audio | Whisper, CLAP, Wav2Vec2 |
| Multimodal | CLIP, LLaVA, BLIP |

## Contributing

Contributions are welcome! Please see our [Contributing Guide](./CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.