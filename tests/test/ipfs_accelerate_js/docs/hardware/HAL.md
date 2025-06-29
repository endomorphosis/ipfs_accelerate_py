# Hardware Abstraction Layer (HAL)

The Hardware Abstraction Layer (HAL) provides a unified interface for accessing hardware acceleration capabilities across different browsers and backend types. It automatically selects the optimal backend based on browser capabilities and model requirements.

## Overview

HAL is designed to simplify the process of using hardware acceleration for AI models in web browsers. It provides:

- **Unified API**: A consistent interface for working with tensors and operations across different backends
- **Automatic Backend Selection**: Selects the best backend based on browser and hardware capabilities
- **Cross-Model Tensor Sharing**: Efficient sharing of tensors between models
- **Browser-Specific Optimization**: Automatically applies optimizations for different browsers
- **Operation Fusion**: Combines multiple operations for better performance
- **Comprehensive Error Handling**: Provides detailed error information and fallback mechanisms

## Usage

### Creating a Hardware Abstraction Layer

```typescript
import { createHardwareAbstractionLayer } from 'ipfs-accelerate-js';

// Create a Hardware Abstraction Layer
const hal = createHardwareAbstractionLayer({
  backends: [],  // Will be populated automatically
  autoInitialize: true,
  useBrowserOptimizations: true,
  enableTensorSharing: true,
  enableOperationFusion: true,
  memoryCacheSize: 100 * 1024 * 1024  // 100MB
});

// Initialize HAL
await hal.initialize();
```

### Creating and Manipulating Tensors

```typescript
// Create a tensor
const tensor = await hal.createTensor({
  dimensions: [2, 3],
  data: new Float32Array([1, 2, 3, 4, 5, 6]),
  dtype: 'float32'
});

// Perform operations
const result = await hal.matmul(tensor, tensor);
const activated = await hal.relu(result);

// Release tensors when done
await hal.releaseTensor(tensor);
await hal.releaseTensor(result);
await hal.releaseTensor(activated);
```

### Working with Neural Network Operations

```typescript
// Feed-forward neural network layer with activation
const output = await hal.ffnn(
  input,
  weights,
  bias,
  'gelu'  // Activation function: 'relu', 'gelu', 'sigmoid', 'tanh', or 'none'
);

// Layer normalization
const normalized = await hal.layerNorm(
  input,
  weight,
  bias,
  1e-12  // Epsilon
);
```

### Using with Cross-Model Tensor Sharing

```typescript
// Create a shared tensor for use by other models
const sharedTensor = hal.createSharedTensor(
  tensor,
  'text_embedding',  // Type of embedding
  'bert-base-uncased'  // Model ID
);

// Get a previously shared tensor
const existingSharedTensor = hal.getSharedTensor(
  'text_embedding',
  'bert-base-uncased'
);

// Use the shared tensor
if (existingSharedTensor) {
  const tensor = existingSharedTensor.tensor;
  // Use the tensor...
}
```

### Backend Information and Synchronization

```typescript
// Get the active backend type
const backendType = hal.getBackendType();  // 'webgpu', 'webnn', or 'cpu'

// Get the active backend
const backend = hal.getActiveBackend();

// Set a specific backend as active
hal.setActiveBackend(customBackend);

// Synchronize backend execution
await hal.sync();

// Clean up resources
hal.dispose();
```

## Integration with Hardware-Abstracted Models

HAL is the foundation for hardware-abstracted model implementations like BERT, Whisper, and CLIP. These models use HAL for all tensor operations and hardware acceleration:

```typescript
import { 
  createHardwareAbstractedModel, 
  StorageManager
} from 'ipfs-accelerate-js';

// Create a storage manager for model weights
const storageManager = {
  // Implement the StorageManager interface
  async initialize() { /* ... */ },
  async getItem(key) { /* ... */ },
  async setItem(key, value) { /* ... */ },
  async hasItem(key) { /* ... */ },
  async removeItem(key) { /* ... */ },
};

// Create a hardware-abstracted BERT model
const bert = createHardwareAbstractedModel('bert', {
  modelId: 'bert-base-uncased',
  backendPreference: ['webgpu', 'webnn', 'cpu'],
  allowFallback: true,
  taskType: 'embedding'
}, storageManager);

// Initialize the model
await bert.initialize();

// Use the model
const result = await bert.predict("This is a test sentence.");
```

## Advanced Configuration

You can configure HAL with various options to optimize for specific use cases:

```typescript
const hal = createHardwareAbstractionLayer({
  // Available backends (will be auto-populated if empty)
  backends: [],
  
  // Default backend to use if no suitable backend is found
  defaultBackend: 'cpu',
  
  // Whether to automatically initialize backends when requested
  autoInitialize: true,
  
  // Whether to use browser-specific optimizations
  useBrowserOptimizations: true,
  
  // Browser type if known, otherwise it will be auto-detected
  browserType: 'chrome',
  
  // Whether to enable tensor sharing between models
  enableTensorSharing: true,
  
  // Whether to enable operation fusion
  enableOperationFusion: true,
  
  // Memory cache size in bytes
  memoryCacheSize: 200 * 1024 * 1024  // 200MB
});
```

## Browser Compatibility

HAL works across all major browsers with varying levels of hardware acceleration support:

| Browser | WebGPU | WebNN | CPU Fallback |
|---------|--------|-------|--------------|
| Chrome  | ✅     | ❌    | ✅           |
| Firefox | ✅     | ❌    | ✅           |
| Edge    | ✅     | ✅    | ✅           |
| Safari  | ✅     | ❌    | ✅           |

## API Reference

### HardwareAbstractionLayer Class

The main class that provides the unified interface for hardware acceleration.

#### Methods

- `initialize()`: Initialize the hardware abstraction layer
- `isInitialized()`: Check if the HAL is initialized
- `getRecommendations(modelType)`: Get hardware recommendations for a model type
- `selectBackend(criteria)`: Select the best backend for given criteria
- `setActiveBackend(backend)`: Set the active backend
- `getActiveBackend()`: Get the active backend
- `getBackendType()`: Get the active backend type
- `createTensor(options)`: Create a tensor on the active backend
- `releaseTensor(tensor)`: Release a tensor from the active backend
- `add(a, b)`: Execute tensor addition
- `subtract(a, b)`: Execute tensor subtraction
- `multiply(a, b)`: Execute tensor multiplication (element-wise)
- `divide(a, b)`: Execute tensor division
- `matmul(a, b, options)`: Execute matrix multiplication
- `transpose(tensor, axes)`: Execute tensor transpose
- `reshape(tensor, newShape)`: Execute tensor reshape
- `gelu(tensor)`: Execute GELU activation function
- `relu(tensor)`: Execute ReLU activation function
- `sigmoid(tensor)`: Execute sigmoid activation function
- `tanh(tensor)`: Execute tanh activation function
- `softmax(tensor, axis)`: Execute softmax activation function
- `layerNorm(input, weight, bias, epsilon)`: Execute layer normalization
- `sqrt(tensor)`: Execute square root operation
- `reduceSum(tensor, axis, keepDims)`: Execute sum reduction
- `reduceMean(tensor, axis, keepDims)`: Execute mean reduction
- `slice(input, starts, ends)`: Slice a tensor
- `concat(tensors, axis)`: Concatenate tensors along an axis
- `gather(input, indices, axis)`: Gather elements from a tensor
- `gatherEmbedding(embedding, indices)`: Gather embedding from a lookup table
- `repeat(input, repeats)`: Repeat a tensor along dimensions
- `ffnn(input, weights, bias, activation)`: Execute FFNN layer with optimizations
- `createSharedTensor(tensor, outputType, modelId)`: Create a shared tensor
- `getSharedTensor(outputType, modelId)`: Get a shared tensor if available
- `sync()`: Synchronize backend execution
- `dispose()`: Dispose of all resources

## Performance Considerations

- **Tensor Creation**: Creating tensors is relatively expensive. Create tensors once and reuse them when possible.
- **Tensor Release**: Always release tensors when you're done with them to avoid memory leaks.
- **Backend Selection**: The automatic backend selection is usually optimal, but you can manually select a backend if needed.
- **Operation Fusion**: Enable operation fusion for better performance on complex neural network operations.
- **Browser Optimizations**: Enable browser-specific optimizations for best performance on each browser.

## Example: Using HAL for Multi-Model Applications

```typescript
// Initialize HAL
const hal = createHardwareAbstractionLayer({
  useBrowserOptimizations: true,
  enableTensorSharing: true
});
await hal.initialize();

// Process text with BERT-like operations
const textTensor = await hal.createTensor({ /* text embedding */ });
const textResult = await hal.ffnn(textTensor, textWeights, textBias, 'gelu');
hal.createSharedTensor(textResult, 'text_embedding', 'my-model');

// Process image with ViT-like operations
const imageTensor = await hal.createTensor({ /* image data */ });
const imageResult = await hal.ffnn(imageTensor, imageWeights, imageBias, 'gelu');
hal.createSharedTensor(imageResult, 'vision_embedding', 'my-model');

// Combine embeddings for multimodal processing
const textEmbedding = hal.getSharedTensor('text_embedding', 'my-model');
const visionEmbedding = hal.getSharedTensor('vision_embedding', 'my-model');

if (textEmbedding && visionEmbedding) {
  // Combine embeddings for multimodal processing
  // ...
}

// Clean up resources
hal.dispose();
```