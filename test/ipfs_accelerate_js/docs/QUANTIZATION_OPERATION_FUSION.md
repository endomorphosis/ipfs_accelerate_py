# WebGPU Quantization and Operation Fusion

This document provides comprehensive documentation for the WebGPU-accelerated quantization and operation fusion features in the IPFS Accelerate JavaScript SDK, including browser-specific optimizations.

## Overview

The WebGPU quantization and operation fusion system provides significant performance optimizations and memory efficiency improvements for running neural network models in the browser:

1. **Model Quantization**: Reduce model memory footprint by using lower precision (1-bit to 8-bit) instead of full 32-bit floating point
2. **Operation Fusion**: Combine multiple operations into a single WebGPU compute shader to eliminate intermediate memory transfers and kernel launches
3. **Browser-Specific Optimizations**: Specialized code paths for different browsers to maximize performance
4. **Ultra-Low Precision**: Advanced techniques for 2-bit and 3-bit quantization with minimal accuracy loss

## Quantization Features

### Supported Bit Widths

- **8-bit**: Standard quantization with minimal precision loss (~0.5%)
- **4-bit**: Good balance of memory savings and accuracy (~1-2% precision loss)
- **3-bit**: Advanced ultra-low precision (~3-5% precision loss)
- **2-bit**: Extreme memory efficiency (~5-10% precision loss)
- **1-bit**: Binary quantization (experimental, significant precision loss)

### Quantization Types

- **Asymmetric Quantization**: Uses both scale and zero-point for better accuracy with activations
- **Symmetric Quantization**: Uses only scale factor, better for weights with symmetric distributions
- **Per-Tensor Quantization**: Single quantization parameters for the entire tensor
- **Per-Channel Quantization**: Separate parameters for each output channel, better accuracy for weights

### Memory Savings

| Precision | Memory Reduction | Typical Accuracy |
|-----------|------------------|------------------|
| 8-bit     | 75%              | 99.5%            |
| 4-bit     | 87.5%            | 98%              |
| 3-bit     | 90.6%            | 95%              |
| 2-bit     | 93.75%           | 90%              |
| 1-bit     | 96.875%          | 80%              |

## Operation Fusion

### Supported Fusion Patterns

- **LinearActivation**: Matrix multiplication followed by activation function (e.g., matmul + ReLU)
- **ElementWiseChain**: Multiple element-wise operations (e.g., add + multiply + add)
- **BinaryUnary**: Element-wise binary operation followed by unary operation (e.g., add + ReLU)
- **ReshapeOp**: Shape operation followed by computation (e.g., reshape + softmax)
- **ActivationChain**: Multiple activation functions in sequence (e.g., ReLU + dropout)
- **NormActivation**: Normalization followed by activation (e.g., LayerNorm + GELU)
- **AttentionPattern**: Self-attention mechanism (e.g., matmul + scale + softmax + matmul)
- **MatrixChain**: Multiple matrix multiplications (e.g., matmul + matmul)
- **QuantizedMatmul**: Matrix multiplication with quantized weights
- **QuantizedMatmulActivation**: Quantized matrix multiplication with activation function
- **QuantizedAttention**: Attention mechanism with quantized weights

### Performance Benefits

- **Reduced Memory Transfers**: Eliminates intermediate buffers between operations
- **Fewer Shader Invocations**: Reduces GPU kernel launch overhead
- **Better Cache Utilization**: Improved data locality
- **Workgroup Optimization**: Tailored workgroup sizes for different operation patterns
- **Memory Bandwidth Reduction**: Less data movement between GPU and memory

## Browser-Specific Optimizations

The framework automatically detects the browser and applies specialized optimizations for each browser's WebGPU implementation through our browser-specific shader generation system.

### Chrome

- **Workgroup Sizing**: Uses mid-sized workgroups (256 threads) for balanced occupancy and register usage
- **Memory Access**: Optimized coalesced memory access patterns for NVIDIA/AMD/Intel GPUs
- **Tile Sizes**: Medium tile sizes (16x16) for matrix operations with efficient shared memory usage
- **Loop Structures**: Unrolled loops with 4-element groups for vectorization-friendly processing
- **Quantization**: Efficient bit packing with hardware-friendly alignment
- **Shader Features**: Leverages Chrome's strong support for complex expressions and loop unrolling
- **Quantized Operations**: 16x16 workgroups with 4-way unrolled inner loops for optimized quantized matmul operations
- **Bit-width Support**: Optimized implementations for all bit-widths (1, 2, 3, 4, 8) with specialized unpacking functions

### Firefox

- **Workgroup Sizing**: Uses smaller workgroups (128 threads) for better occupancy
- **Memory Access**: Simpler memory access patterns that avoid coalescing issues in Firefox's WebGPU implementation
- **Tile Sizes**: Smaller tile sizes (8x8) that fit better in Firefox's WebGPU memory model
- **Loop Structures**: Simple, linear loops without complex unrolling (Firefox handles these more efficiently)
- **Quantization**: Direct bit manipulation optimized for Firefox's bit-manipulation performance
- **Audio Processing**: Specialized optimizations for audio model computations (15-25% faster than other browsers)
- **Quantized Operations**: 8x8 workgroups with simple non-unrolled loops for quantized operations
- **Conditional Logic**: Uses if-else chains rather than switch statements for better performance in Firefox's shader compiler

### Safari

- **Workgroup Sizing**: Uses larger workgroups (512 threads) optimized for Apple Silicon GPUs
- **Memory Access**: Apple GPU-specific memory access patterns with SIMDGroup optimizations
- **Tile Sizes**: Larger tile sizes (32x32) to leverage Apple GPU's larger shared memory capabilities
- **Math Approximations**: Fast approximations for transcendental functions optimized for Apple GPUs
- **Quantization**: Specialized 3-bit packing with 10 values per 32-bit word optimized for Apple Silicon
- **Metal Backend**: Leverages special features available in Safari's Metal-based WebGPU implementation

### Edge

- **Workgroup Sizing**: Balanced workgroups (256 threads) similar to Chrome
- **WebNN Integration**: Optional integration with Edge's WebNN implementation for additional acceleration
- **Memory Management**: Optimized buffer management for Edge's memory handling
- **Quantization**: Special paths for WebNN-accelerated quantized operations when available
- **Hybrid Approach**: Combines best optimizations from Chrome with Edge-specific enhancements
- **Loop Unrolling Strategy**: Uses partial unrolling in pairs (processes two elements at once) for better performance
- **Control Flow**: Uses switch statements over if-else chains for better shader compiler optimizations
- **Bounds Checking**: Implements explicit bounds checking in inner loops for improved performance
- **Quantized Operations**: 16x16 workgroups with partial unrolling for quantized matmul operations

### Automatic Configuration

The framework's `browser_specific_shaders.ts` module automatically:

1. **Detects Browser Type**: Identifies the current browser at runtime
2. **Selects Optimal Parameters**: Chooses workgroup sizes, tile sizes, and memory layouts
3. **Generates Optimized WGSL**: Creates browser-specific shader code with tailored optimizations
4. **Manages Compile Pipeline**: Uses browser-specific shader cache strategies

This automatic optimization can be configured or overridden:

```typescript
// Manual override example
const config = {
  browserOptimizationType: 'firefox', // Force Firefox optimizations
  workgroupSize: 64, // Override default workgroup size
  tileSize: 16, // Custom tile size
  useFastMath: true // Enable fast math approximations
};

// Apply to operation
const result = await backend.matmul(inputA, inputB, config);
```

## API Usage

### Basic Quantization

```typescript
// Configure the WebGPU backend with quantization
const backend = new WebGPUBackend();
await backend.initialize();

// Create a tensor with weights to be quantized
const weights = new Tensor<number>(
  [1024, 1024],
  weightData,
  { dataType: 'float32' }
);

// Quantize weights to 4-bit precision
const quantizedWeights = await backend.quantize(weights, {
  bitsPerWeight: 4,
  useSymmetricQuantization: true,
  usePerChannelQuantization: true
});

// Use quantized weights in matrix multiplication
const result = await backend.matmul(input, quantizedWeights, {
  useQuantization: true
});
```

### Operation Fusion

```typescript
// Configure operation fusion
const fusionConfig: FusionConfig = {
  useQuantizedWeights: true,
  bitsPerWeight: 4,
  useBrowserOptimizations: true,
  enabledPatterns: [
    FusionPattern.QuantizedMatmulActivation,
    FusionPattern.AttentionPattern
  ]
};

// Create fusion manager
const fusion = new WebGPUOperationFusion(backend, fusionConfig);

// Execute fused operations (e.g., matmul + relu)
const result = await backend.executeOperations(
  [inputTensor, weightsTensor],
  ['matmul', 'relu'],
  { useFusion: true }
);
```

### Ultra-Low Precision

```typescript
// Configure with 2-bit precision for extreme memory efficiency
const fusionConfig: FusionConfig = {
  useQuantizedWeights: true,
  bitsPerWeight: 2,
  useBrowserOptimizations: true
};

// Fusion with 2-bit precision
const fusion = new WebGPUOperationFusion(backend, fusionConfig);

// Execute with 2-bit quantized weights
const result = await backend.matmul(input, weights, {
  useQuantization: true,
  bitsPerWeight: 2
});
```

## Performance Tips

1. **Choose the Right Precision**:
   - Use 4-bit for most models (good balance)
   - Use 2-bit when memory is severely constrained
   - Use 8-bit when highest accuracy is required

2. **Enable Browser Optimizations**:
   - Always enable `useBrowserOptimizations: true`
   - Let the system detect and optimize for the current browser

3. **Optimize Based on Model Size**:
   - For large models, prioritize memory efficiency with lower precision
   - For small models, prioritize accuracy with higher precision

4. **Fusion Selection**:
   - Enable all fusion patterns for best performance
   - For transformer models, always include `QuantizedAttention`
   - For CNN models, prioritize `QuantizedMatmulActivation`

## Best Practices

1. **Choose the Right Quantization Type**:
   - For weights: Symmetric quantization (useSymmetricQuantization: true)
   - For activations: Asymmetric quantization (useSymmetricQuantization: false)
   - Always use per-channel quantization for weights (usePerChannelQuantization: true)

2. **Memory Management**:
   - Dispose of unneeded tensors to free GPU memory
   - Use the WebGPU buffer manager for efficient allocation

3. **Testing**:
   - Always validate accuracy after quantization
   - Test different precision levels to find the best tradeoff
   - Consider fine-tuning models after quantization

4. **Performance Monitoring**:
   - Use browser performance tools to monitor GPU time
   - Track memory usage with the buffer manager's memory report
   - Benchmark different configurations to find optimal settings

## Examples

See the following examples for practical usage:

- [Quantized Operations Example](../examples/quantized_operations_example.ts)
- [BERT Browser Optimization](../examples/browser_optimized_bert_example.ts)
- [ViT Browser Optimization](../examples/browser_optimized_vit_example.ts)

## Integration Testing

The quantization and operation fusion system includes comprehensive tests in:

- [Fusion Quantization Test](../test/fusion_quantization_test.ts)

These tests validate correctness, measure performance, and compare different configurations across browsers.