# WebNN Graph Building Guide

## Overview

The WebNN Graph Builder is a high-level API for building and executing neural network graphs using the W3C Web Neural Network API (WebNN). It provides a convenient interface for constructing complex neural networks with hardware acceleration on supported browsers.

This implementation is part of the IPFS Accelerate TypeScript SDK, enabling efficient neural network inference in web browsers with WebNN support.

## Key Features

- **Graph-based execution**: Build neural network graphs with optimized execution
- **Hardware acceleration**: Leverage browser WebNN implementations for GPU/NPU acceleration
- **High-level API**: Create neural networks with minimal code
- **Graph caching**: Optimize performance with graph reuse
- **Operation fusion**: Automatically combine operations for better performance
- **Neural network operations**: Convolution, pooling, normalization, matrix operations, etc.
- **Common layer patterns**: Full Neural Network layers, residual blocks, batch normalization, etc.

## Prerequisites

- A browser with WebNN support (Chrome 113+, Edge 113+, Safari 17.0+)
- TypeScript 4.9+ for development

## Usage Examples

### Basic Graph Building

```typescript
// Initialize the backend and graph builder
const backend = new WebNNBackend();
await backend.initialize();

const graphBuilder = new WebNNGraphBuilder(backend);
await graphBuilder.initialize();

// Create input tensor
const inputTensor = new Tensor([1, 10], Array(10).fill(0.5), { dataType: 'float32' });

// Create input node
const inputNode = graphBuilder.input('input', inputTensor);

// Create a hidden layer with ReLU activation
const hiddenWeights = new Tensor([10, 20], Array(10 * 20).fill(0.1), { dataType: 'float32' });
const hiddenBias = new Tensor([20], Array(20).fill(0.1), { dataType: 'float32' });
const hiddenNode = graphBuilder.layer(inputNode, hiddenWeights, hiddenBias, 'relu');

// Create an output layer
const outputWeights = new Tensor([20, 5], Array(20 * 5).fill(0.1), { dataType: 'float32' });
const outputBias = new Tensor([5], Array(5).fill(0.1), { dataType: 'float32' });
const outputNode = graphBuilder.layer(hiddenNode, outputWeights, outputBias, 'none');

// Apply softmax
const probsNode = graphBuilder.softmax(outputNode);
graphBuilder.output('output', probsNode);

// Execute the graph
const results = await graphBuilder.execute({ 'input': inputTensor });
console.log('Output:', results.output.data);

// Cleanup
graphBuilder.dispose();
backend.dispose();
```

### Sequential Model

```typescript
// Create a sequential model with multiple layers
const model = graphBuilder.sequential(inputNode, [
  {
    // Hidden layer 1: 10 -> 20 with ReLU activation
    weights: new Tensor([10, 20], Array(10 * 20).fill(0.1), { dataType: 'float32' }),
    bias: new Tensor([20], Array(20).fill(0.1), { dataType: 'float32' }),
    activation: 'relu'
  },
  {
    // Hidden layer 2: 20 -> 20 with ReLU activation
    weights: new Tensor([20, 20], Array(20 * 20).fill(0.1), { dataType: 'float32' }),
    bias: new Tensor([20], Array(20).fill(0.1), { dataType: 'float32' }),
    activation: 'relu'
  },
  {
    // Output layer: 20 -> 5 with sigmoid activation
    weights: new Tensor([20, 5], Array(20 * 5).fill(0.1), { dataType: 'float32' }),
    bias: new Tensor([5], Array(5).fill(0.1), { dataType: 'float32' }),
    activation: 'sigmoid'
  }
]);

graphBuilder.output('output', model);
```

### Convolutional Neural Network (CNN)

```typescript
// Input: [1, 28, 28, 1] (MNIST format: batch, height, width, channels)
const inputNode = graphBuilder.input('input', inputTensor);

// First conv layer: 3x3 filter, 16 output channels, ReLU activation
const conv1Filter = graphBuilder.constant(filterTensor);
const conv1 = graphBuilder.conv2d(inputNode, conv1Filter, {
  strides: [1, 1],
  padding: [1, 1, 1, 1], // Same padding
  activation: 'relu'
});

// Max pooling: 2x2 window, stride 2
const pool1 = graphBuilder.pool(conv1, 'max', {
  windowDimensions: [2, 2],
  strides: [2, 2]
});

// Second conv layer
const conv2Filter = graphBuilder.constant(filter2Tensor);
const conv2 = graphBuilder.conv2d(pool1, conv2Filter, {
  strides: [1, 1],
  padding: [1, 1, 1, 1],
  activation: 'relu'
});

// Flatten to [1, 1568]
const flattened = graphBuilder.reshape(conv2, [1, 1568]);

// Fully connected layer
const fc = graphBuilder.layer(flattened, fcWeights, fcBias, 'relu');

// Output layer with softmax
const probsNode = graphBuilder.softmax(fc);
graphBuilder.output('output', probsNode);
```

### Residual Block

```typescript
// Create a residual block with two fully connected layers
const residualBlock = graphBuilder.residualBlock(
  inputNode,
  layer1Weights,
  layer1Bias,
  layer2Weights,
  layer2Bias,
  'relu'
);
```

### Graph Caching

```typescript
// First execution builds and caches the graph
await graphBuilder.executeWithCache({ 'input': tensor1 }, 'inference_graph');

// Subsequent executions reuse the cached graph
await graphBuilder.executeWithCache({ 'input': tensor2 }, 'inference_graph');
```

## API Reference

### WebNNGraphBuilder

```typescript
class WebNNGraphBuilder {
  // Constructor
  constructor(backend: WebNNBackend);
  
  // Initialization
  async initialize(): Promise<void>;
  
  // Graph building
  input(name: string, tensor: Tensor<any>): GraphNode;
  constant(tensor: Tensor<any>): GraphNode;
  output(name: string, node: GraphNode): void;
  
  // Graph execution
  async buildAndCompile(): Promise<any>;
  async execute(inputs: Record<string, Tensor<any>>, graph?: any): Promise<Record<string, Tensor<any>>>;
  async executeWithCache(inputs: Record<string, Tensor<any>>, cacheKey: string): Promise<Record<string, Tensor<any>>>;
  
  // Neural network operations
  conv2d(input: GraphNode, filter: GraphNode, options?: ConvolutionOptions): GraphNode;
  pool(input: GraphNode, type: 'max' | 'average', options: PoolingOptions): GraphNode;
  matmul(a: GraphNode, b: GraphNode): GraphNode;
  gemm(a: GraphNode, b: GraphNode, options?: GemmOptions): GraphNode;
  batchNormalization(input: GraphNode, mean: GraphNode, variance: GraphNode, scale?: GraphNode, bias?: GraphNode, epsilon?: number): GraphNode;
  
  // Element-wise operations
  add(a: GraphNode, b: GraphNode): GraphNode;
  sub(a: GraphNode, b: GraphNode): GraphNode;
  mul(a: GraphNode, b: GraphNode): GraphNode;
  div(a: GraphNode, b: GraphNode): GraphNode;
  
  // Tensor manipulation
  reshape(input: GraphNode, newShape: number[]): GraphNode;
  transpose(input: GraphNode, permutation?: number[]): GraphNode;
  concat(inputs: GraphNode[], axis: number): GraphNode;
  
  // Activation functions
  activation(input: GraphNode, activation: ActivationFunction): GraphNode;
  softmax(input: GraphNode, axis?: number): GraphNode;
  
  // Higher-level layer patterns
  fullyConnected(input: GraphNode, weights: GraphNode, bias?: GraphNode, activation?: ActivationFunction): GraphNode;
  layer(input: GraphNode, weights: Tensor<any>, bias?: Tensor<any>, activation?: ActivationFunction): GraphNode;
  residualBlock(input: GraphNode, fc1Weights: Tensor<any>, fc1Bias: Tensor<any>, fc2Weights: Tensor<any>, fc2Bias: Tensor<any>, activation?: ActivationFunction): GraphNode;
  sequential(input: GraphNode, layers: Array<{ weights: Tensor<any>; bias?: Tensor<any>; activation?: ActivationFunction; }>): GraphNode;
  
  // Resource management
  reset(): void;
  clearCache(): void;
  dispose(): void;
}
```

### Interfaces

```typescript
interface ConvolutionOptions {
  strides?: [number, number];
  padding?: [number, number, number, number] | number;
  dilations?: [number, number];
  groups?: number;
  layout?: 'nchw' | 'nhwc';
  activation?: ActivationFunction;
}

interface PoolingOptions {
  windowDimensions: [number, number];
  strides?: [number, number];
  padding?: [number, number, number, number] | number;
  dilations?: [number, number];
  layout?: 'nchw' | 'nhwc';
}

interface GemmOptions {
  alpha?: number;
  beta?: number;
  aTranspose?: boolean;
  bTranspose?: boolean;
  c?: Tensor<any>;
}

type ActivationFunction = 'relu' | 'sigmoid' | 'tanh' | 'leakyRelu' | 'none';
```

## Optimizations

The WebNN Graph Builder includes several optimizations:

1. **Graph Caching**: Compiled graphs are cached to avoid recompilation
2. **Operation Fusion**: Compatible operations are fused when possible
3. **Memory Management**: Efficient tensor allocation and deallocation
4. **Browser-Specific Optimizations**: Adapts to different browser WebNN implementations
5. **Compute/Memory Overlap**: Minimizes data movement between CPU and GPU

## Browser Compatibility

WebNN support varies by browser:

| Browser | Version | Features | Notes |
|---------|---------|----------|-------|
| Chrome  | 113+    | Basic WebNN, Conv2D, Pooling, MatMul | Best overall WebNN support |
| Edge    | 113+    | Basic WebNN, Conv2D, Pooling, MatMul | Similar to Chrome |
| Safari  | 17.0+   | Basic WebNN, limited operations | Early implementation |
| Firefox | 113+*   | Experimental, limited operations | Enable in about:config |

*Firefox requires enabling `dom.webnn.enabled` in about:config

## Performance Considerations

- **Graph Building**: Building graphs can be expensive; use caching for repeated inference
- **Data Types**: WebNN primarily supports float32; quantization is handled separately
- **Memory Usage**: Large models may require chunked execution or model sharding
- **Browser Variations**: Performance can vary significantly between browsers and hardware
- **Power Efficiency**: Set the power preference based on the task requirements

## Integration with Other Components

The WebNN Graph Builder works well with other components in the IPFS Accelerate SDK:

- **WebGPU Backend**: For more control over compute shaders when WebNN is not available
- **Tensor Sharing System**: Efficiently share tensors between models 
- **Storage Manager**: Cache model weights and tensors in IndexedDB
- **Hardware Detector**: Automatically select optimal backend based on capabilities

## Resources

- [W3C WebNN Specification](https://www.w3.org/TR/webnn/)
- [WebNN API Explainer](https://github.com/webmachinelearning/webnn/blob/main/explainer.md)
- [Chrome WebNN Status](https://www.chromestatus.com/feature/5650817042432000)
- [WebNN Polyfill](https://github.com/webmachinelearning/webnn-polyfill) for unsupported browsers