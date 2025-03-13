# IPFS Accelerate JavaScript SDK

A hardware-accelerated machine learning framework for browsers using WebGPU and WebNN.

## Features

- **Tensor Operations**: Efficient tensor operations with TypeScript typing
- **Cross-Model Tensor Sharing**: Memory optimization through tensor reuse and sharing
- **Hardware Acceleration**: WebGPU and WebNN backends for accelerated computation
- **Flexible Storage**: CPU, WebGPU, and WebNN tensor storage options
- **Memory Management**: Reference counting for efficient tensor lifecycle management
- **GPU-Accelerated Operations**: Matrix multiplication, element-wise operations, and neural network functions
- **Browser Detection**: Intelligent hardware capability detection and optimization

## Installation

```bash
npm install ipfs-accelerate
```

## Quick Start

### Basic Tensor Operations

```typescript
import { Tensor, ones, random } from 'ipfs-accelerate';

// Create a basic tensor
const tensor = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
console.log(tensor.toString());

// Create matrices for multiplication
const matrixA = random([128, 256], -1, 1);
const matrixB = ones([256, 128]);

// Perform operations
const result = matrixA.matmul(matrixB);
console.log(result.shape); // [128, 128]
```

### WebGPU Acceleration

```typescript
import { WebGPUBackend, random } from 'ipfs-accelerate';

// Initialize WebGPU backend
const backend = new WebGPUBackend();
await backend.initialize();

// Create WebGPU-accelerated tensors
const matrixA = random([1024, 1024], -1, 1, { backend: 'webgpu' });
const matrixB = random([1024, 1024], -1, 1, { backend: 'webgpu' });

// Perform GPU-accelerated matrix multiplication
const result = await backend.matmul(matrixA, matrixB);

// Perform other operations
const activated = await backend.relu(result);
console.log(activated.shape); // [1024, 1024]

// Cleanup
backend.dispose();
```

### Cross-Model Tensor Sharing

```typescript
import { TensorSharingManager } from 'ipfs-accelerate';

// Create a tensor sharing manager
const manager = new TensorSharingManager(1024); // 1GB max memory

// Register a shared tensor
const embedding = manager.registerSharedTensor(
  "bert_embedding",
  [1, 768],
  "webgpu",
  "bert-base",
  ["t5", "gpt2"],
  "float32"
);

// Optimize memory usage
const result = manager.optimizeMemoryUsage();
console.log(`Memory reduction: ${result.memory_reduction_percent}%`);
```

## Development

### Prerequisites

- Node.js 16+
- npm or yarn
- Browser with WebGPU support (Chrome 113+, Edge 113+, Safari 17.4+)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ipfs-accelerate.git
cd ipfs-accelerate

# Install dependencies
npm install

# Build the project
npm run build
```

### Running Examples

```bash
# Build and serve examples
npm run start:examples

# Open WebGPU tensor example
# http://localhost:8080/dist/examples/webgpu_tensor_example.html

# Open Tensor sharing example
# http://localhost:8080/dist/examples/tensor_sharing_example.html
```

### Testing

```bash
# Run tests
npm test
```

## WebGPU Acceleration

The SDK provides hardware acceleration using WebGPU for significant performance improvements:

| Operation | CPU Time | WebGPU Time | Speedup |
|-----------|----------|-------------|---------|
| Matrix Multiplication (1024x1024) | ~2000ms | ~50ms | ~40x |
| Element-wise Operations | ~15ms | ~2ms | ~7.5x |
| ReLU Activation | ~12ms | ~2ms | ~6x |
| Sigmoid Activation | ~25ms | ~3ms | ~8x |

WebGPU acceleration includes:
- **Efficient Buffer Management**: Reuses GPU buffers to minimize allocations
- **WGSL Shader Collection**: Optimized compute shaders for tensor operations
- **Asynchronous Execution**: Non-blocking operation execution
- **Workgroup Optimization**: Optimized workgroup sizes for different operations

## Documentation

- [API Reference](./docs/api/README.md)
- [Examples](./src/examples/)
- [WebGPU Implementation Summary](../WEBGPU_IMPLEMENTATION_SUMMARY.md)
- [WebGPU Next Steps](../WEBGPU_NEXT_STEPS.md)

## Key Components

### Tensor Module

- `Tensor`: Base tensor class with generic typing
- `SharedTensor`: Tensor with reference counting for cross-model sharing
- `TensorSharingManager`: Manager for tensor registration and sharing
- `Operations`: Matrix, arithmetic, and neural network operations

### Hardware Module

- `HardwareBackend`: Common interface for all hardware backends
- `WebGPUBackend`: Hardware acceleration using WebGPU (Completed)
- `WebNNBackend`: Neural network acceleration with WebNN (Planned)
- `HardwareDetector`: Browser capability detection and optimization

### WebGPU Module

- `WebGPUBufferManager`: GPU memory management with buffer reuse
- `Shaders`: WGSL compute shader collection for tensor operations

## Browser Compatibility

| Browser | Minimum Version | WebGPU Support |
|---------|----------------|----------------|
| Chrome  | 113+           | ✅ Full Support |
| Edge    | 113+           | ✅ Full Support |
| Firefox | 117+           | ⚠️ Experimental |
| Safari  | 17.4+          | ✅ Full Support |

## License

MIT