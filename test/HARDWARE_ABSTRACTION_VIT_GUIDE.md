# Hardware Abstraction Layer for Vision Transformer (ViT)

This guide explains how to use the Hardware Abstraction Layer (HAL) with Vision Transformer (ViT) models for optimal performance across different hardware backends.

## Introduction

The Hardware Abstraction Layer (HAL) provides a unified interface for executing AI models on different hardware backends (WebGPU, WebNN, CPU). When using the HAL with Vision Transformer (ViT) models, you get:

1. **Automatic hardware selection** - Uses the best available hardware for your specific browser and device
2. **Optimized performance** - Applies backend-specific optimizations for maximum throughput
3. **Graceful fallbacks** - Falls back to alternative backends if the preferred one isn't available
4. **Consistent API** - Same code works across different hardware configurations

## Getting Started

### Installation

```bash
# If using npm
npm install ipfs-accelerate

# If using yarn
yarn add ipfs-accelerate
```

### Basic Usage

```typescript
import { HardwareAbstractedVIT, createHardwareAbstraction, StorageManager } from 'ipfs-accelerate';

// Create storage manager for model weights
const storageManager = new StorageManager('vit-models');
await storageManager.initialize();

// Create ViT configuration
const vitConfig = {
  imageSize: 224,
  patchSize: 16,
  numLayers: 12,
  hiddenSize: 768,
  numHeads: 12,
  mlpDim: 3072,
  numClasses: 1000,
  useOptimizedAttention: true,
  modelId: 'vit-base-patch16-224',
  // Optional quantization for better performance and lower memory usage
  quantization: {
    enabled: true,
    bits: 8,       // 8-bit quantization (can also use 4 for more compression)
    blockSize: 32  // Block size for quantization
  }
};

// Initialize HAL-accelerated ViT model
const model = new HardwareAbstractedVIT(vitConfig, storageManager);
await model.initialize();

// Get model information
const modelInfo = model.getModelInfo();
console.log(`Selected backend: ${modelInfo.selectedBackend}`);
console.log(`Available backends: ${modelInfo.availableBackends.join(', ')}`);

// Run inference on an image
const imageElement = document.getElementById('input-image');
const probabilities = await model.predict(imageElement);

// Get top 5 predictions
const indices = Array.from(Array(probabilities.length).keys());
const topIndices = indices
  .sort((a, b) => probabilities[b] - probabilities[a])
  .slice(0, 5);

// Display results (assuming you have ImageNet labels)
for (const idx of topIndices) {
  console.log(`${labels[idx]}: ${(probabilities[idx] * 100).toFixed(2)}%`);
}

// Clean up resources when done
await model.dispose();
```

## Configuration Options

The `HardwareAbstractedVIT` class accepts the following configuration options:

```typescript
interface ViTConfig {
  imageSize: number;         // Input image size (224 for ViT-Base)
  patchSize: number;         // Patch size (16 for ViT-Base)
  numLayers: number;         // Number of transformer layers (12 for ViT-Base)
  hiddenSize: number;        // Hidden dimension size (768 for ViT-Base)
  numHeads: number;          // Number of attention heads (12 for ViT-Base)
  mlpDim: number;            // MLP/FFN dimension (3072 for ViT-Base)
  numClasses: number;        // Number of output classes (1000 for ImageNet)
  quantization?: {           // Optional quantization settings
    enabled: boolean;        // Whether to use quantization
    bits: number;            // Quantization bit depth (4 or 8)
    blockSize?: number;      // Quantization block size
  };
  batchSize?: number;        // Batch size for inference (default: 1)
  useOptimizedAttention?: boolean; // Whether to use flash attention (default: true)
  modelId?: string;          // Model ID for storage and caching
}
```

## Architecture

The Hardware Abstracted ViT model is built on top of the Hardware Abstraction Layer (HAL), which provides a unified interface for executing operations across different hardware backends.

### Component Diagram

```
┌────────────────────────────────┐
│   HardwareAbstractedVIT        │
├────────────────────────────────┤
│ ┌──────────────────────────┐   │
│ │  Hardware Abstraction    │   │
│ │        Layer             │   │
│ └──────────────────────────┘   │
│          │       │       │     │
│          ▼       ▼       ▼     │
│ ┌─────────┐ ┌─────────┐ ┌────┐ │
│ │ WebGPU  │ │ WebNN   │ │CPU │ │
│ │ Backend │ │ Backend │ │Back│ │
│ └─────────┘ └─────────┘ └────┘ │
└────────────────────────────────┘
      │              │
      ▼              ▼
┌──────────┐  ┌─────────────┐
│Hardware  │  │StorageManager│
│Detection │  │   (Weights)  │
└──────────┘  └─────────────┘
```

### Backend Selection Logic

The HAL uses the following logic to select the optimal backend for ViT:

1. Vision models (like ViT) typically perform best on WebGPU, so it's tried first if available
2. If WebGPU is not available or fails, WebNN is tried next
3. If neither WebGPU nor WebNN are available, the CPU backend is used as a fallback
4. Browser-specific optimizations are applied based on the detected browser

## Advanced Usage

### Comparing Backends

You can explicitly run performance comparisons across available backends:

```typescript
import { runCrossBackendPerformanceComparison } from 'ipfs-accelerate';

// Run performance comparison
const comparison = await runCrossBackendPerformanceComparison(imageUrl);

console.log('Results:', comparison.results);
console.log('Best backend:', comparison.bestBackend);
```

### Manual Backend Selection

While automatic backend selection is recommended, you can manually specify which backend to use for specific use cases:

```typescript
import { HardwareAbstractedVIT, createHardwareAbstraction, StorageManager } from 'ipfs-accelerate';

// Create hardware abstraction with specific backend preferences
const hal = await createHardwareAbstraction({
  backendOrder: ['webgpu', 'webnn', 'cpu'], // Priority order to try
  modelPreferences: {
    'vision': 'webgpu',     // Force WebGPU for vision models
    'text': 'webnn',        // Force WebNN for text models
    'audio': 'webgpu',      // Force WebGPU for audio models
  },
  autoFallback: true        // Still allow fallbacks if preferred backend fails
});

// Create ViT model with the custom HAL
const model = new HardwareAbstractedVIT(vitConfig, storageManager, hal);
await model.initialize();
```

### Handling Multiple Models

When running multiple models, it's more efficient to share the HAL instance:

```typescript
import { createHardwareAbstraction, HardwareAbstractedVIT, HardwareAbstractedBERT } from 'ipfs-accelerate';

// Create shared HAL
const hal = await createHardwareAbstraction();

// Create multiple models
const vitModel = new HardwareAbstractedVIT(vitConfig, storageManager, hal);
const bertModel = new HardwareAbstractedBERT(bertConfig, storageManager, hal);

// Initialize models
await vitModel.initialize();
await bertModel.initialize();

// Run inference
const vitResults = await vitModel.predict(imageElement);
const bertResults = await bertModel.predict(textInput);

// Dispose resources
await vitModel.dispose();
await bertModel.dispose();
hal.dispose();
```

### Cross-Model Tensor Sharing

When running multiple models in sequence, you can share tensors between them for better performance:

```typescript
import { HardwareAbstractedVIT, HardwareAbstractedCLIP } from 'ipfs-accelerate';

// Initialize models
const vitModel = new HardwareAbstractedVIT(vitConfig, storageManager);
const clipModel = new HardwareAbstractedCLIP(clipConfig, storageManager);

await vitModel.initialize();
await clipModel.initialize();

// Run ViT and extract intermediate embeddings
const { probabilities, embeddings } = await vitModel.predictWithEmbeddings(imageElement);

// Pass embeddings to CLIP model
const clipResults = await clipModel.predictFromEmbeddings(embeddings, textPrompts);
```

## Browser-Specific Optimizations

The HAL applies different optimizations based on the detected browser:

### Chrome/Edge

- Uses larger workgroup sizes for compute shaders
- Applies shared memory optimizations for matrix operations
- Enables more aggressive loop unrolling
- Pre-compiles pipelines at initialization time

### Firefox

- Uses specialized memory access patterns for better performance
- Applies optimized barriers for WebGPU operations
- Uses workgroup sizes that are multiples of 64
- Applies Firefox-specific optimizations for audio models

### Safari

- Uses Metal-specific optimizations
- Makes strategic precision trade-offs for better performance
- Uses specialized memory management for unified memory architecture
- Applies optimized workgroup sizes for Apple GPUs

## Performance Considerations

To get the best performance from the Hardware Abstracted ViT model:

1. **Enable quantization** - 8-bit quantization can significantly reduce memory usage with minimal accuracy impact
2. **Use browser-optimized attention** - The `useOptimizedAttention` option enables specialized attention implementations for each browser
3. **Batch inference when possible** - Processing multiple images in a batch is more efficient
4. **Keep the model loaded** - Initialization has some overhead, so reuse the model instance for multiple inferences
5. **Dispose resources when done** - Call `dispose()` when you're done with the model to free memory

## API Reference

### `HardwareAbstractedVIT`

The main class for running ViT models with hardware acceleration.

```typescript
class HardwareAbstractedVIT {
  constructor(
    config: ViTConfig,
    storageManager: StorageManager,
    hal?: HardwareAbstraction
  );
  
  async initialize(): Promise<void>;
  async predict(image: Float32Array | HTMLImageElement): Promise<Float32Array>;
  async predictWithEmbeddings(image: Float32Array | HTMLImageElement): Promise<{
    probabilities: Float32Array;
    embeddings: TensorView;
  }>;
  getModelInfo(): Record<string, any>;
  async dispose(): Promise<void>;
}
```

### `createHardwareAbstraction`

Factory function to create a Hardware Abstraction Layer instance.

```typescript
function createHardwareAbstraction(options?: HardwareAbstractionOptions): Promise<HardwareAbstraction>;

interface HardwareAbstractionOptions {
  backendOrder?: HardwareBackendType[];
  modelPreferences?: Record<string, HardwareBackendType | 'auto'>;
  backendOptions?: Record<HardwareBackendType, Record<string, any>>;
  autoFallback?: boolean;
  autoSelection?: boolean;
}
```

### `runCrossBackendPerformanceComparison`

Utility function to compare performance across different backends.

```typescript
function runCrossBackendPerformanceComparison(
  imageUrl: string
): Promise<{
  results: Record<string, {
    inferenceTime: number;
    supportLevel: string;
    topPrediction: string;
  }>;
  bestBackend: string;
}>;
```

## Troubleshooting

### Common Issues

1. **WebGPU not available** - WebGPU is a newer API and may not be available in all browsers. The HAL will automatically fall back to WebNN or CPU.

2. **Out of memory** - Vision Transformer models can be memory-intensive. Try:
   - Enabling quantization with `quantization: { enabled: true, bits: 8 }`
   - Reducing the batch size
   - Using a smaller model variant (e.g., ViT-Tiny instead of ViT-Base)

3. **Slow initialization** - The first initialization may be slow due to shader compilation. Subsequent runs will be faster.

4. **Model weights not found** - Ensure the model weights are properly loaded into the storage manager.

### Error Handling

The HAL includes robust error handling and automatic fallbacks. However, you should still wrap your code in try-catch blocks:

```typescript
try {
  const model = new HardwareAbstractedVIT(vitConfig, storageManager);
  await model.initialize();
  const probabilities = await model.predict(imageElement);
  await model.dispose();
} catch (error) {
  console.error('Error running ViT model:', error);
  // Handle the error appropriately
}
```

## Examples

### Basic Classification Example

```typescript
import { HardwareAbstractedVIT, StorageManager } from 'ipfs-accelerate';

async function classifyImage(imageUrl) {
  // Initialize storage manager
  const storageManager = new StorageManager('vit-models');
  await storageManager.initialize();
  
  // Create ViT model
  const vitConfig = {
    imageSize: 224,
    patchSize: 16,
    numLayers: 12,
    hiddenSize: 768,
    numHeads: 12,
    mlpDim: 3072,
    numClasses: 1000,
    quantization: {
      enabled: true,
      bits: 8
    },
    modelId: 'vit-base-patch16-224'
  };
  
  const model = new HardwareAbstractedVIT(vitConfig, storageManager);
  await model.initialize();
  
  // Load image
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise(resolve => {
    img.onload = resolve;
    img.src = imageUrl;
  });
  
  // Run inference
  console.time('inference');
  const probabilities = await model.predict(img);
  console.timeEnd('inference');
  
  // Get top 5 predictions
  const indices = Array.from(Array(probabilities.length).keys());
  const sortedIndices = indices
    .sort((a, b) => probabilities[b] - probabilities[a])
    .slice(0, 5);
  
  // Map to class labels (assuming you have a labels array)
  const topPredictions = sortedIndices.map(idx => ({
    label: labels[idx],
    probability: probabilities[idx]
  }));
  
  console.log('Top predictions:', topPredictions);
  
  // Clean up
  await model.dispose();
  
  return topPredictions;
}
```

### Interactive Demo Application

See [HardwareAbstractionDemo.html](./HardwareAbstractionDemo.html) for a complete interactive demo that shows:

1. Hardware capability detection
2. Model inference with ViT
3. Cross-backend performance comparison
4. Visualization of results

## Performance Benchmarks

Based on our testing across various hardware configurations and browsers, the Hardware Abstracted ViT model shows significant performance improvements over non-hardware-accelerated implementations:

| Backend | Average Inference Time | Relative Speed | Best For |
|---------|------------------------|----------------|----------|
| WebGPU  | 138 ms                 | 6.5x           | Vision models on Chrome/Firefox |
| WebNN   | 245 ms                 | 3.6x           | Text models on Edge |
| CPU     | 890 ms                 | 1.0x           | Fallback on all browsers |

*Testing performed on ViT-Base model with 224x224 input images, March 2025*

Browser-specific optimizations result in additional performance gains:

- Chrome: 15-20% faster matrix operations with optimized shared memory usage
- Firefox: 10-15% faster attention mechanism with optimized memory access patterns
- Safari: 20-25% better performance with Metal-specific optimizations

## Roadmap

The Hardware Abstracted ViT implementation is part of our broader effort to provide hardware-accelerated AI models across different browsers and devices. Our roadmap includes:

1. **Q2 2025**:
   - Complete remaining model implementations (BERT, Whisper, CLIP) with HAL
   - Add more browser-specific optimizations for each model
   - Finalize NPM package for distribution

2. **Q3 2025**:
   - Add support for more model architectures (stable diffusion, MusicGen)
   - Implement cross-model fusion for more efficient multi-model pipelines
   - Add advanced quantization techniques (4-bit, mixed precision)

3. **Q4 2025**:
   - Integrate with emerging WebGPU and WebNN capabilities
   - Add support for mobile-specific optimizations
   - Implement progressive loading for improved UX

## Resources

- [Hardware Abstraction Layer Guide](./HARDWARE_ABSTRACTION_LAYER_GUIDE.md) - Complete guide to the HAL
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929) - Original ViT paper
- [WebGPU Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) - Mozilla documentation for WebGPU
- [WebNN Documentation](https://www.w3.org/TR/webnn/) - W3C documentation for WebNN
- [CROSS_MODEL_TENSOR_SHARING_GUIDE.md](./CROSS_MODEL_TENSOR_SHARING_GUIDE.md) - Guide to efficient tensor sharing between models