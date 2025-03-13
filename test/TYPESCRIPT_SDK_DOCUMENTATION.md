# IPFS Accelerate JavaScript SDK Documentation

**Current Version:** 0.8.0-alpha  
**Updated:** March 16, 2025  
**Status:** In Development (80% complete)

## Overview

The IPFS Accelerate JavaScript SDK provides optimized machine learning inference capabilities in the browser using WebNN and WebGPU hardware acceleration. It enables efficient AI model execution with features such as cross-model tensor sharing, model weight caching, and adaptive hardware selection.

## Key Features

- **Hardware Acceleration**: Leverages WebNN and WebGPU for optimal performance across browsers
- **Model Compatibility**: Supports popular model architectures like ViT, BERT, Whisper, and more
- **Model Weight Caching**: Persistent storage of model weights using IndexedDB
- **Adaptive Hardware Selection**: Automatically selects the best available hardware backend
- **Cross-Browser Optimization**: Specialized optimizations for Chrome, Firefox, Edge, and Safari
- **Tensor Operations**: Comprehensive suite of tensor operations with hardware acceleration
- **Memory Efficiency**: Intelligent memory management with garbage collection

## Installation (Planned for release)

```bash
npm install ipfs-accelerate
```

## Quick Start Example

```javascript
import { WebNNBackend, WebNNStorageIntegration } from 'ipfs-accelerate';

async function runInference() {
  // Create and initialize WebNN backend
  const backend = new WebNNBackend();
  await backend.initialize();
  
  // Create storage integration
  const storage = new WebNNStorageIntegration(backend);
  await storage.initialize();
  
  // Check if model is already cached
  const modelId = 'bert-base-uncased';
  if (await storage.isModelCached(modelId)) {
    // Load model from cache
    const modelTensors = await storage.loadModel(modelId);
    
    // Create input tensor
    const inputIds = new Int32Array([101, 2054, 2003, 2026, 2171, 1029, 102]);
    const inputTensor = await backend.createTensor(
      inputIds,
      [1, 7],
      'int32'
    );
    
    // Run inference
    const result = await backend.execute('bert', {
      input_ids: inputTensor,
      weights: modelTensors
    });
    
    // Get output
    const outputData = await backend.readTensor(result.tensor, result.shape);
    console.log('Inference result:', outputData);
  } else {
    console.log('Model not cached. Please download and store it first.');
  }
}

runInference();
```

## Architecture

The SDK follows a modular architecture with several key components:

### Hardware Abstraction Layer (HAL)

Provides a consistent interface for different hardware backends:

```typescript
interface HardwareBackend {
  type: HardwareBackendType;
  isSupported(): Promise<boolean>;
  initialize(): Promise<boolean>;
  getCapabilities(): Promise<Record<string, any>>;
  createTensor(data: TypedArray, shape: number[], dataType?: string): Promise<any>;
  execute(operation: string, inputs: Record<string, any>, options?: Record<string, any>): Promise<any>;
  readTensor(tensor: any, shape: number[], dataType?: string): Promise<TypedArray>;
  dispose(): void;
}
```

### WebNN Backend

Implements the Hardware Abstraction Layer using the WebNN API for neural network acceleration:

```typescript
const backend = new WebNNBackend({
  deviceType: 'gpu',               // 'gpu' or 'cpu'
  powerPreference: 'high-performance', // 'high-performance', 'low-power', or 'default'
  enableLogging: true,             // Enable logging for debugging
  floatPrecision: 'float32',       // 'float32' or 'float16'
  preferSyncExecution: true,       // Use sync execution when available
  memory: {                       
    enableGarbageCollection: true, // Enable garbage collection of unused tensors
    garbageCollectionThreshold: 1024 * 1024 * 128 // 128MB threshold
  }
});

await backend.initialize();
```

### WebNN Standalone Interface

Provides an easier-to-use interface for WebNN capabilities without requiring the full HAL:

```typescript
import { 
  isWebNNSupported, 
  getWebNNDeviceInfo, 
  getWebNNBrowserRecommendations,
  runWebNNExample
} from 'ipfs-accelerate/webnn-standalone';

// Check if WebNN is supported
const supported = await isWebNNSupported();

// Get device information
const deviceInfo = await getWebNNDeviceInfo();

// Get browser recommendations
const recommendations = getWebNNBrowserRecommendations();

// Run a simple example operation
const result = await runWebNNExample('matmul');
```

### Storage Manager

Provides persistent storage for model weights and tensors using IndexedDB:

```typescript
const storageManager = new StorageManager({
  dbName: 'my-model-cache',        // Name of the IndexedDB database
  enableCompression: true,         // Enable compression for stored tensors
  maxStorageSize: 1024 * 1024 * 1024, // 1GB maximum storage size
  enableAutoCleanup: true,         // Enable automatic cleanup of unused items
  cleanupThreshold: 1000 * 60 * 60 * 24 * 7, // 7 days threshold
  enableLogging: true              // Enable logging for debugging
});

await storageManager.initialize();
```

### WebNN Storage Integration

Integrates the WebNN backend with the Storage Manager for efficient model loading:

```typescript
const storage = new WebNNStorageIntegration(backend, {
  enableModelCaching: true,        // Enable model weights caching
  enableAutoCleanup: true,         // Enable automatic cleanup of unused models
  maxStorageSize: 1024 * 1024 * 1024, // 1GB maximum storage size
  dbName: 'webnn-model-cache'      // Database name
});

await storage.initialize();

// Store a model
await storage.storeModel(
  'my-model',                      // Model ID
  'My Model',                      // Model name
  tensorMap,                       // Map of tensor names to tensor data
  { version: '1.0.0' }             // Optional metadata
);

// Load a model
const modelTensors = await storage.loadModel('my-model');
```

## Supported Operations

The SDK supports the following neural network operations with hardware acceleration:

### Core Operations
- **Matrix Multiplication**: Multiply two matrices
- **Convolution**: 2D convolution for convolutional neural networks
- **Softmax**: Softmax activation function
- **Elementwise**: Basic elementwise operations (relu, sigmoid, tanh)

### Pooling Operations
- **Max Pooling**: Maximum value in a sliding window
- **Average Pooling**: Average value in a sliding window

### Normalization Operations
- **Batch Normalization**: Normalize across the batch dimension
- **Layer Normalization**: Normalize across specified axes

### Advanced Elementwise Operations
- **Add**: Element-wise addition
- **Subtract**: Element-wise subtraction
- **Multiply**: Element-wise multiplication
- **Divide**: Element-wise division
- **Power**: Element-wise exponentiation
- **Minimum**: Element-wise minimum
- **Maximum**: Element-wise maximum
- **Exponential**: Element-wise exponential function
- **Logarithm**: Element-wise natural logarithm
- **Square Root**: Element-wise square root

### Tensor Manipulation Operations
- **Reshape**: Change tensor shape without altering data
- **Transpose**: Permute tensor dimensions
- **Concatenate**: Join tensors along a specified axis
- **Slice**: Extract a slice from a tensor
- **Pad**: Pad a tensor with a constant value

## Browser Compatibility

| Browser | WebNN Support | WebGPU Support | Recommended For |
|---------|---------------|----------------|-----------------|
| Edge    | Excellent     | Good           | WebNN operations, general use |
| Chrome  | Good          | Excellent      | WebGPU operations, vision models |
| Firefox | Limited       | Good           | Audio models, compute shaders |
| Safari  | Experimental  | Good           | Vision models on Apple Silicon |

## Performance Considerations

### Device Selection
Choose the appropriate device type for your workload:
```typescript
const backend = new WebNNBackend({
  deviceType: 'gpu', // Use GPU for most neural network operations
  powerPreference: 'high-performance' // Prioritize performance over battery life
});
```

### Memory Management
Explicitly dispose of tensors when no longer needed:
```typescript
// After you're done with a tensor
backend.dispose();
```

### Model Caching
Use the storage manager to cache models for improved loading times:
```typescript
// Store the model once
await storage.storeModel('model-id', 'Model Name', tensorMap);

// Then load it efficiently in future sessions
const modelTensors = await storage.loadModel('model-id');
```

## Error Handling

The SDK uses async/await patterns for most operations, so use try/catch for error handling:

```javascript
try {
  const result = await backend.execute('matmul', {
    a: inputTensor,
    b: weightsTensor
  });
  
  const output = await backend.readTensor(result.tensor, result.shape);
  console.log('Success:', output);
} catch (error) {
  console.error('Inference failed:', error);
  
  // Handle specific error types
  if (error.name === 'NotSupportedError') {
    console.error('This operation is not supported on your device');
  } else if (error.name === 'OutOfMemoryError') {
    console.error('Not enough memory to run this model');
  }
}
```

## Examples

### Basic Tensor Operations

```javascript
// Create input tensors
const tensorA = await backend.createTensor(
  new Float32Array([1, 2, 3, 4]),
  [2, 2],
  'float32'
);

const tensorB = await backend.createTensor(
  new Float32Array([5, 6, 7, 8]),
  [2, 2],
  'float32'
);

// Matrix multiplication
const result = await backend.execute('matmul', {
  a: tensorA,
  b: tensorB
});

// Read the result
const output = await backend.readTensor(result.tensor, result.shape);
console.log('Result:', Array.from(output));
```

### Image Classification with ViT

```javascript
// Load the ViT model from storage
const modelTensors = await storage.loadModel('vit-base-patch16-224');

// Preprocess the image
const imageData = await preprocessImage(imageElement, 224, 224);
const inputTensor = await backend.createTensor(
  imageData,
  [1, 224, 224, 3],
  'float32'
);

// Run inference
const result = await backend.execute('vit', {
  input: inputTensor,
  weights: modelTensors
});

// Get class predictions
const logits = await backend.readTensor(result.tensor, result.shape);
const predictions = getTopClasses(logits, classLabels, 5);
console.log('Top predictions:', predictions);
```

### Text Embedding with BERT

```javascript
// Load BERT model from storage
const modelTensors = await storage.loadModel('bert-base-uncased');

// Tokenize input text
const tokens = tokenizer.encode('Hello, world!');
const inputTensor = await backend.createTensor(
  new Int32Array(tokens),
  [1, tokens.length],
  'int32'
);

// Run inference
const result = await backend.execute('bert', {
  input_ids: inputTensor,
  weights: modelTensors
});

// Get embedding
const embedding = await backend.readTensor(result.tensor, result.shape);
console.log('Text embedding:', embedding);
```

## Current Status and Roadmap

The SDK is currently in development with approximately 80% of features implemented. The following components are complete:

- ✅ WebNN backend implementation
- ✅ WebNN standalone interface
- ✅ Additional WebNN operations (pooling, normalization, elementwise, manipulation)
- ✅ Storage Manager for model weights with IndexedDB
- ✅ WebNN Storage Integration

Upcoming features (planned for April-May 2025):

- 🔲 Operation fusion for better performance
- 🔲 WebGPU compute shader operations
- 🔲 WGSL shader implementations for core tensor operations
- 🔲 Cross-model tensor sharing with reference counting
- 🔲 Browser-specific shader optimizations
- 🔲 Model implementations (ViT, BERT, Whisper)
- 🔲 Comprehensive examples for different model types
- 🔲 NPM package preparation for release

## Contributing

This SDK is currently in internal development. The public release and contribution guidelines will be available when the SDK reaches beta status, expected in June 2025.

## License

To be determined upon public release.

## Acknowledgments

This SDK builds upon the work of numerous open-source projects and standards, including:

- WebNN API specification
- WebGPU API specification
- TensorFlow.js
- ONNX Runtime Web