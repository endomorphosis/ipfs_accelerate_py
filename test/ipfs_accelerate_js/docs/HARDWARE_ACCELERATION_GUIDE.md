# Hardware Acceleration Guide for Web ML

This guide explains how to leverage hardware acceleration for machine learning in web browsers using the IPFS Accelerate JavaScript SDK.

## Table of Contents

- [Introduction](#introduction)
- [Hardware Acceleration Technologies](#hardware-acceleration-technologies)
- [Browser Support](#browser-support)
- [Optimizing for Different Hardware](#optimizing-for-different-hardware)
- [Performance Considerations](#performance-considerations)
- [Debugging Hardware Acceleration](#debugging-hardware-acceleration)
- [Best Practices](#best-practices)

## Introduction

Hardware acceleration can provide significant performance improvements for machine learning workloads in web browsers. The IPFS Accelerate SDK supports two primary hardware acceleration technologies:

1. **WebGPU**: A modern API for GPU compute, providing general-purpose computing capabilities
2. **WebNN**: A neural network-specific API designed for hardware acceleration of ML models

By using these technologies, the SDK can achieve substantial speedups (up to 40x for certain operations) compared to CPU-based implementations.

## Hardware Acceleration Technologies

### WebGPU

WebGPU is a modern graphics and compute API for the web. For machine learning, it provides:

- **General-purpose compute** via compute shaders
- **High-performance matrix operations**
- **Flexible memory management**
- **Fine-grained control** over execution

WebGPU is well-suited for:
- Matrix multiplications and linear algebra
- Custom operations and algorithms
- Large-scale tensor operations

### WebNN

WebNN (Web Neural Network) is an API specifically designed for neural network acceleration. It provides:

- **Graph-based neural network execution**
- **Transparent hardware acceleration** (including NPUs, DSPs, GPUs)
- **Optimized neural network operations**
- **Power-efficient execution**

WebNN is ideal for:
- Neural network inference
- Activation functions and convolutions
- Energy-efficient execution on mobile devices
- Utilizing specialized ML hardware (like Apple Neural Engine)

### Performance Comparison

| Operation | CPU | WebGPU | WebNN | Best Option |
|-----------|-----|--------|-------|-------------|
| Matrix Multiplication (1024x1024) | ~2000ms | ~50ms | ~75ms | WebGPU |
| Element-wise Operations | ~15ms | ~2ms | ~2.5ms | WebGPU |
| ReLU Activation | ~12ms | ~2ms | ~1.5ms | WebNN |
| Sigmoid Activation | ~25ms | ~3ms | ~2ms | WebNN |
| Tanh Activation | ~24ms | ~3.5ms | ~2.3ms | WebNN |

## Browser Support

### WebGPU Support

| Browser | Version | Support Level | Notes |
|---------|---------|--------------|-------|
| Chrome  | 113+    | ✅ Excellent  | Stable implementation with good performance |
| Edge    | 113+    | ✅ Excellent  | Based on Chromium, similar to Chrome |
| Safari  | 17.4+   | ✅ Good       | Full support with some performance differences |
| Firefox | 117+    | ✅ Good       | Initially behind flag, now enabled by default |

### WebNN Support

| Browser | Version | Support Level | Notes |
|---------|---------|--------------|-------|
| Chrome  | 113+    | ✅ Good       | Early but functional implementation |
| Edge    | 113+    | ✅ Good       | Good integration with DirectML |
| Safari  | 17.4+   | ✅ Excellent  | Best performance on Apple Silicon |
| Firefox | 117+    | ⚠️ Experimental | Behind flag, limited support |

### Apple Silicon (M1/M2/M3)

Apple Silicon devices deserve special mention because:

- They have dedicated Neural Engine (NPU) hardware
- Safari provides direct access to this hardware through WebNN
- Performance for neural network operations can be 2-5x better than WebGPU on these devices

## Optimizing for Different Hardware

### Desktop GPUs

```typescript
import { createOptimalBackend } from 'ipfs-accelerate';

// For desktop GPUs, prefer WebGPU
const backend = await createOptimalBackend({
  backend: 'webgpu',
  preferSpeed: true
});
```

### Apple Silicon Devices

```typescript
import { createOptimalBackend, hasNeuralProcessor } from 'ipfs-accelerate';

// Check for Apple Neural Engine
const hasNPU = hasNeuralProcessor();

// If Apple Silicon, let the system pick (likely WebNN)
const backend = await createOptimalBackend({
  preferSpeed: true
});
```

### Mobile Devices

```typescript
import { createOptimalBackend } from 'ipfs-accelerate';

// For mobile, balance power and performance
const backend = await createOptimalBackend({
  preferLowPower: true
});
```

### Multi-Backend Applications

For applications that need to work optimally across all devices:

```typescript
import { createMultiBackend } from 'ipfs-accelerate';

// Try WebNN first, then WebGPU, then CPU
const backend = await createMultiBackend(['webnn', 'webgpu', 'cpu']);
```

## Performance Considerations

### Memory Management

Both WebGPU and WebNN have memory limitations:

```typescript
// For WebGPU, enable garbage collection
if (backend.type === 'webgpu') {
  // Cast to WebGPUBackend for specific methods
  const webgpuBackend = backend as WebGPUBackend;
  
  // Clean up unused buffers periodically
  setInterval(() => {
    webgpuBackend.garbageCollect();
  }, 5000);
}

// Always dispose of tensors when done
tensor.dispose();
```

### Operation Batching

For WebNN, batching operations into a single graph can significantly improve performance:

```typescript
// Inefficient - each operation creates a new graph
const result1 = await backend.relu(tensor);
const result2 = await backend.sigmoid(result1);

// More efficient - use the multi-backend feature to optimize the graph
const graph = await createGraph(backend);
const input = graph.input(tensor);
const hidden = graph.relu(input);
const output = graph.sigmoid(hidden);
const result = await graph.execute(output);
```

### Asynchronous Execution

All hardware-accelerated operations are asynchronous:

```typescript
// Process tensors in parallel
const promises = tensors.map(tensor => backend.relu(tensor));
const results = await Promise.all(promises);
```

## Debugging Hardware Acceleration

### Checking Capabilities

```typescript
import { detectHardware, detectWebNNFeatures } from 'ipfs-accelerate';

// Get general hardware information
const hardware = await detectHardware();
console.log('WebGPU available:', hardware.hasWebGPU);
console.log('WebNN available:', hardware.hasWebNN);

// Get WebNN-specific information
const webnnFeatures = await detectWebNNFeatures();
console.log('WebNN hardware accelerated:', webnnFeatures.hardwareAccelerated);
console.log('Acceleration type:', webnnFeatures.accelerationType);
```

### Performance Testing

```typescript
import { WebGPUBackend, WebNNBackend, random } from 'ipfs-accelerate';

// Test WebGPU performance
const gpuBackend = new WebGPUBackend();
await gpuBackend.initialize();

// Test WebNN performance
const nnBackend = new WebNNBackend();
await nnBackend.initialize();

// Create test tensor
const tensor = random([1024, 1024], -1, 1);

// WebGPU performance test
console.time('webgpu-matmul');
await gpuBackend.matmul(tensor, tensor);
console.timeEnd('webgpu-matmul');

// WebNN performance test
console.time('webnn-matmul');
await nnBackend.matmul(tensor, tensor);
console.timeEnd('webnn-matmul');
```

## Best Practices

### Hardware Selection

1. **Use the optimal backend for each operation type**:
   - WebGPU for general tensor operations and matrix multiplications
   - WebNN for neural network operations (activations, convolutions)

2. **Implement fallbacks for best cross-browser support**:
   ```typescript
   const backend = await createMultiBackend(['webnn', 'webgpu', 'cpu']);
   ```

3. **Consider device-specific optimizations**:
   - Apple Silicon: Prefer WebNN for neural network operations
   - Mobile devices: Consider power efficiency
   - High-performance desktops: Maximize WebGPU utilization

### Performance Optimization

1. **Reduce memory transfers**:
   - Keep tensors on the same backend when possible
   - Use the SharedTensor functionality for memory optimization

2. **Batch operations for WebNN**:
   - Combine multiple operations into a single graph
   - Use the graph API for complex operation sequences

3. **Reuse resources**:
   - Cache commonly used tensors
   - Reuse buffers and graphs when possible

4. **Be mindful of browser limits**:
   - Monitor memory usage
   - Release resources when not needed
   - Implement garbage collection for long-running applications

### Browser-Specific Considerations

1. **Chrome/Edge**:
   - Generally balanced performance for both WebGPU and WebNN
   - Good for applications that need both types of acceleration

2. **Safari**:
   - Exceptional WebNN performance on Apple Silicon
   - Use WebNN for neural network operations on Apple devices

3. **Firefox**:
   - Prioritize WebGPU as WebNN support is experimental
   - Implement solid fallbacks

By following these guidelines, you can effectively leverage hardware acceleration for machine learning in web browsers across a wide range of devices and browsers.