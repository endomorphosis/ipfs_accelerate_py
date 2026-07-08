# Hardware Abstraction Layer (HAL) Guide

**Date: March 14, 2025**  
**Status: Completed**

The Hardware Abstraction Layer (HAL) provides a unified interface for executing tensor operations across different hardware backends including WebGPU, WebNN, and CPU. This guide explains how to use the HAL, its architecture, and best practices.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Architecture](#architecture)
4. [Backend Selection](#backend-selection)
5. [Executing Operations](#executing-operations)
6. [Error Handling and Fallbacks](#error-handling-and-fallbacks)
7. [Advanced Usage](#advanced-usage)
8. [Browser Compatibility](#browser-compatibility)
9. [Examples](#examples)
10. [API Reference](#api-reference)

## Introduction

The HAL abstracts away the complexities of working with different hardware acceleration APIs in browsers. It allows you to execute the same code regardless of the underlying hardware, automatically selecting the most appropriate backend based on hardware capabilities and model type.

Key features:
- Unified interface for WebGPU, WebNN, and CPU backends
- Automatic hardware capability detection
- Intelligent backend selection based on model type
- Automatic fallback when preferred backends are unavailable
- Comprehensive error handling

## Getting Started

### Installation

```typescript
// Import the HAL from the IPFS Accelerate package
import { createHardwareAbstraction } from 'ipfs-accelerate';

// Create and initialize the HAL
const hal = await createHardwareAbstraction();

// Execute operations
const result = await hal.execute('matmul', {
  a: { data: floatArrayA, shape: [2, 3] },
  b: { data: floatArrayB, shape: [3, 2] }
});

// Clean up resources when done
hal.dispose();
```

### Basic Example

Here's a simple example of using the HAL to execute a matrix multiplication:

```typescript
import { createHardwareAbstraction } from 'ipfs-accelerate';

async function runMatrixMultiplication() {
  // Create sample matrices
  const matrixA = new Float32Array([1, 2, 3, 4]);
  const matrixB = new Float32Array([5, 6, 7, 8]);
  
  // Create and initialize HAL
  const hal = await createHardwareAbstraction();
  
  try {
    // Execute matrix multiplication
    const result = await hal.execute('matmul', {
      a: { data: matrixA, shape: [2, 2] },
      b: { data: matrixB, shape: [2, 2] }
    });
    
    console.log('Result shape:', result.shape);
    console.log('Result data:', new Float32Array(result.data));
  } finally {
    // Always clean up resources
    hal.dispose();
  }
}
```

## Architecture

The HAL consists of the following components:

1. **HardwareAbstraction**: The main class that provides the unified interface
2. **HardwareBackend**: Interface implemented by all backends (WebGPU, WebNN, CPU)
3. **Hardware Detection**: Detects available hardware capabilities
4. **Backend Selection**: Chooses the most appropriate backend for each operation
5. **Operation Execution**: Routes operations to the selected backend

### Backend Implementations

- **WebGPUBackend**: Accelerates operations using WebGPU compute shaders
- **WebNNBackend**: Utilizes the WebNN API for optimized neural network operations
- **CPUBackend**: Provides JavaScript implementations for all operations as a fallback

## Backend Selection

The HAL uses a sophisticated algorithm to select the most appropriate backend for each operation based on:

1. Hardware capabilities
2. Model type
3. Operation characteristics
4. User preferences

### Default Selection Logic

By default, the HAL selects backends in the following order of preference:

1. WebGPU (preferred for vision and audio models)
2. WebNN (preferred for text models)
3. CPU (fallback when other backends are unavailable)

### Model Type Optimization

Different model types perform better on different backends:

| Model Type | Preferred Backend | Reason |
|------------|-------------------|--------|
| Vision     | WebGPU            | Excels at parallel processing for vision models |
| Audio      | WebGPU (Firefox)  | Firefox's compute shader performance is optimal for audio |
| Text       | WebNN             | Better for text models due to optimized matrix operations |
| Generic    | WebGPU            | Good all-around performance |

### Customizing Backend Selection

You can customize backend selection by providing options when creating the HAL:

```typescript
const hal = await createHardwareAbstraction({
  // Custom backend order
  backendOrder: ['webnn', 'webgpu', 'cpu'],
  
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
  }
});
```

## Executing Operations

The HAL supports a wide range of operations including:

### Core Operations

- `matmul`: Matrix multiplication
- `conv2d`: 2D convolution
- `add`, `sub`, `mul`, `div`: Binary operations
- `relu`, `sigmoid`, `tanh`: Activation functions
- `softmax`: Softmax operation

### Tensor Manipulation

- `reshape`: Change tensor shape
- `transpose`: Transpose tensor dimensions
- `concat`: Concatenate tensors
- `slice`: Slice a tensor

### Advanced Operations

- `quantize`: Quantize float32 to int8
- `dequantize`: Dequantize int8 to float32
- `pooling`: Max pooling and average pooling
- `exp`, `log`, `sqrt`: Unary operations

### Executing an Operation

To execute an operation:

```typescript
const result = await hal.execute(
  operation,  // Operation name
  inputs,     // Input tensors
  options     // Operation-specific options
);
```

Example with matrix multiplication:

```typescript
const result = await hal.execute(
  'matmul',
  {
    a: { data: matrixA, shape: [2, 3] },
    b: { data: matrixB, shape: [3, 2] }
  },
  {
    modelType: 'vision',
    transposeA: false,
    transposeB: false
  }
);
```

## Error Handling and Fallbacks

The HAL includes built-in error handling and fallback mechanisms:

### Automatic Fallbacks

If a preferred backend fails to execute an operation, the HAL can automatically fall back to the next available backend:

```typescript
const hal = await createHardwareAbstraction({
  autoFallback: true  // Enable automatic fallbacks (default: true)
});
```

### Manual Error Handling

For more control, you can handle errors manually:

```typescript
try {
  const result = await hal.execute('matmul', inputs, {
    preferredBackend: 'webgpu'  // Try using WebGPU
  });
} catch (error) {
  console.error('WebGPU execution failed:', error);
  
  // Fall back to CPU manually
  const fallbackResult = await hal.execute('matmul', inputs, {
    preferredBackend: 'cpu'
  });
}
```

## Advanced Usage

### Getting Hardware Capabilities

You can get detailed information about available hardware:

```typescript
const capabilities = hal.getCapabilities();
console.log('WebGPU supported:', capabilities.webgpuSupported);
console.log('WebNN supported:', capabilities.webnnSupported);
console.log('Browser:', capabilities.browserName, capabilities.browserVersion);
console.log('Platform:', capabilities.platform);
```

### Directly Accessing Backends

For advanced use cases, you can access backends directly:

```typescript
// Check if a specific backend is available
if (hal.hasBackend('webgpu')) {
  // Get the WebGPU backend
  const webgpuBackend = hal.getBackend('webgpu');
  
  // Use backend-specific features
  const capabilities = await webgpuBackend.getCapabilities();
  console.log('WebGPU features:', capabilities.features);
}
```

### Backend-Specific Options

You can pass backend-specific options when executing operations:

```typescript
const result = await hal.execute('matmul', inputs, {
  backendOptions: {
    // WebGPU-specific options
    workgroupSize: [8, 8, 1],
    useSharedMemory: true
  }
});
```

## Browser Compatibility

The HAL supports all major browsers with varying levels of hardware acceleration:

| Browser | WebGPU | WebNN | Best For |
|---------|--------|-------|----------|
| Chrome  | ✅     | ⚠️ (Limited) | Vision models, general use |
| Edge    | ✅     | ✅     | Text models, WebNN acceleration |
| Firefox | ✅     | ❌     | Audio models (excellent compute shaders) |
| Safari  | ⚠️ (Limited) | ❌ | Basic acceleration with fallbacks |

- ✅ Full support
- ⚠️ Limited support
- ❌ No support

## Examples

### Basic Matrix Multiplication

```typescript
import { createHardwareAbstraction } from 'ipfs-accelerate';

async function runMatrixMultiplication() {
  const matrixA = new Float32Array([1, 2, 3, 4, 5, 6]);
  const matrixB = new Float32Array([7, 8, 9, 10, 11, 12]);
  
  const hal = await createHardwareAbstraction();
  
  try {
    const result = await hal.execute('matmul', {
      a: { data: matrixA, shape: [2, 3] },
      b: { data: matrixB, shape: [3, 2] }
    });
    
    console.log('Result shape:', result.shape);
    console.log('Result data:', new Float32Array(result.data));
  } finally {
    hal.dispose();
  }
}
```

### Running Inference for a Vision Model

```typescript
import { createHardwareAbstraction } from 'ipfs-accelerate';

async function runVisionModelInference(inputImage) {
  const hal = await createHardwareAbstraction();
  
  try {
    // Convert image to tensor
    const inputTensor = await convertImageToTensor(inputImage);
    
    // Execute convolution operation for feature extraction
    const features = await hal.execute('conv2d', {
      input: inputTensor,
      filter: convWeights
    }, { modelType: 'vision' });
    
    // Apply ReLU activation
    const activated = await hal.execute('elementwise', {
      input: features
    }, { operation: 'relu', modelType: 'vision' });
    
    // Apply pooling
    const pooled = await hal.execute('maxpool', {
      input: activated
    }, { windowDimensions: [2, 2], modelType: 'vision' });
    
    // Final classification layer
    const logits = await hal.execute('matmul', {
      a: { data: pooled.data, shape: pooled.shape },
      b: { data: classifierWeights, shape: [pooled.shape[1], numClasses] }
    }, { modelType: 'vision' });
    
    // Apply softmax for probabilities
    const probabilities = await hal.execute('softmax', {
      input: logits
    }, { axis: 1, modelType: 'vision' });
    
    return probabilities;
  } finally {
    hal.dispose();
  }
}
```

### Hardware-Specific Benchmarking

```typescript
import { createHardwareAbstraction } from 'ipfs-accelerate';

async function benchmarkBackends() {
  const hal = await createHardwareAbstraction();
  const backends = hal.getAvailableBackends();
  
  const matrixA = new Float32Array(1024 * 1024).fill(1);
  const matrixB = new Float32Array(1024 * 1024).fill(1);
  
  console.log('Available backends:', backends);
  
  for (const backend of backends) {
    console.log(`Benchmarking ${backend}...`);
    
    const startTime = performance.now();
    
    await hal.execute('matmul', {
      a: { data: matrixA, shape: [1024, 1024] },
      b: { data: matrixB, shape: [1024, 1024] }
    }, { preferredBackend: backend });
    
    const endTime = performance.now();
    console.log(`${backend} execution time: ${endTime - startTime}ms`);
  }
  
  hal.dispose();
}
```

## API Reference

### HardwareAbstraction Class

```typescript
class HardwareAbstraction {
  // Constructor
  constructor(options?: HardwareAbstractionOptions);
  
  // Core methods
  async initialize(): Promise<boolean>;
  async execute<T = any, R = any>(operation: string, inputs: T, options?: ExecuteOptions): Promise<R>;
  dispose(): void;
  
  // Backend management
  getBestBackend(modelType: string): HardwareBackend | null;
  getBackend(type: HardwareBackendType): HardwareBackend | null;
  hasBackend(type: HardwareBackendType): boolean;
  getAvailableBackends(): HardwareBackendType[];
  
  // Hardware information
  getCapabilities(): HardwareCapabilities | null;
}
```

### HardwareAbstractionOptions Interface

```typescript
interface HardwareAbstractionOptions {
  // Backend preferences
  backendOrder?: HardwareBackendType[];
  modelPreferences?: Record<string, HardwareBackendType | 'auto'>;
  
  // Backend-specific options
  backendOptions?: Record<HardwareBackendType, Record<string, any>>;
  
  // Behavior options
  autoFallback?: boolean;
  autoSelection?: boolean;
}
```

### ExecuteOptions Interface

```typescript
interface ExecuteOptions {
  // Operation context
  modelType?: string;
  preferredBackend?: HardwareBackendType;
  
  // Backend-specific options
  backendOptions?: Record<string, any>;
}
```

### HardwareBackendType Type

```typescript
type HardwareBackendType = 'webgpu' | 'webnn' | 'wasm' | 'cpu';
```

### Factory Function

```typescript
async function createHardwareAbstraction(
  options?: HardwareAbstractionOptions
): Promise<HardwareAbstraction>;
```

## Conclusion

The Hardware Abstraction Layer provides a unified, easy-to-use interface for executing tensor operations across different hardware backends. By abstracting away the complexities of different APIs and providing intelligent backend selection, it allows you to focus on your application logic while ensuring optimal performance across a wide range of devices and browsers.

For more information, see the [TypeScript SDK Documentation](README_IPFS_ACCELERATE_IMPLEMENTATION.md) and explore the [example applications](examples/).

## Related Documentation

- [WebGPU Backend Documentation](WEBGPU_IMPLEMENTATION_SUMMARY.md)
- [WebNN Backend Documentation](WEBNN_IMPLEMENTATION_SUMMARY.md)
- [Cross-Model Tensor Sharing Guide](CROSS_MODEL_TENSOR_SHARING_GUIDE.md)
- [Browser Optimization Guide](BROWSER_OPTIMIZATION_GUIDE.md)

## Hardware-Abstracted Model Implementations

The HAL is used to implement hardware-accelerated versions of popular machine learning models. These implementations automatically select the optimal backend and apply browser-specific optimizations:

### BERT (Text Model)

The Hardware Abstracted BERT implementation provides efficient text processing with automatic backend selection:

- [HARDWARE_ABSTRACTION_BERT_GUIDE.md](HARDWARE_ABSTRACTION_BERT_GUIDE.md) - Comprehensive guide to the HAL-based BERT implementation
- Best backend: WebNN (Edge) or WebGPU
- Tasks: Text embeddings, sentiment analysis, question answering
- 5.8x speedup vs. CPU on optimal hardware

### Whisper (Audio Model)

The Hardware Abstracted Whisper implementation provides efficient audio transcription and translation:

- [HARDWARE_ABSTRACTION_WHISPER_GUIDE.md](HARDWARE_ABSTRACTION_WHISPER_GUIDE.md) - Comprehensive guide to the HAL-based Whisper implementation
- Best backend: WebGPU (Firefox)
- Tasks: Audio transcription, translation
- 3.0x speedup vs. CPU on optimal hardware
- Special optimizations for Firefox's compute shader performance

### ViT (Vision Model)

The Hardware Abstracted ViT implementation provides efficient image processing:

- [HARDWARE_ABSTRACTION_VIT_GUIDE.md](HARDWARE_ABSTRACTION_VIT_GUIDE.md) - Comprehensive guide to the HAL-based ViT implementation
- Best backend: WebGPU (Chrome)
- Tasks: Image classification, visual embeddings
- 6.5x speedup vs. CPU on optimal hardware
- Browser-specific optimizations for matrix operations

### Multimodal Integration

These hardware-abstracted models can be combined for efficient multimodal applications through cross-model tensor sharing:

```typescript
// Create models with tensor sharing enabled
const vitModel = createHardwareAbstractedViT(hal, { enableTensorSharing: true });
const bertModel = createHardwareAbstractedBERT(hal, { enableTensorSharing: true });

// Process an image with ViT
const imageResult = await vitModel.process(imageData);

// Get the vision embedding
const visionEmbedding = vitModel.getSharedTensor('vision_embedding');

// Process text with BERT
const textResult = await bertModel.predict(text);

// Get the text embedding
const textEmbedding = bertModel.getSharedTensor('text_embedding');

// Use both embeddings for multimodal tasks
// (e.g., image-text matching, visual question answering)
```

For interactive examples, see:
- [hardware_abstracted_bert_example.html](ipfs_accelerate_js/examples/browser/models/hardware_abstracted_bert_example.html)
- [hardware_abstracted_whisper_example.html](ipfs_accelerate_js/examples/browser/models/hardware_abstracted_whisper_example.html)
- [hardware_abstracted_vit_example.html](ipfs_accelerate_js/examples/browser/models/hardware_abstracted_vit_example.html)