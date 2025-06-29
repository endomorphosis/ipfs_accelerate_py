# Ultra-Low Precision Quantization Guide

This guide explores advanced techniques for ultra-low precision quantization (3-bit, 2-bit, and 1-bit) in the IPFS Accelerate JavaScript SDK, enabling extreme memory efficiency with minimal accuracy loss.

## Introduction

Ultra-low precision quantization reduces the memory footprint of neural network models by representing weights with fewer bits than traditional approaches:

- **Traditional quantization**: 8-bit quantization (75% memory reduction)  
- **Standard reduced precision**: 4-bit quantization (87.5% memory reduction)
- **Ultra-low precision**: 3-bit (90.6%), 2-bit (93.75%), and 1-bit (96.875%) quantization

These techniques are especially valuable for:
- Mobile and memory-constrained devices
- Large models that wouldn't otherwise fit in browser memory
- Applications requiring multiple models loaded simultaneously
- Edge deployments with limited resources

## Precision Levels

| Precision | Bits per Weight | Values per 32-bit Word | Memory Reduction | Typical Accuracy Loss |
|-----------|----------------|------------------------|------------------|---------------------|
| FP32      | 32             | 1                      | 0%               | 0% (baseline)       |
| INT8      | 8              | 4                      | 75%              | 0.5%                |
| INT4      | 4              | 8                      | 87.5%            | 1-2%                |
| INT3      | 3              | 10                     | 90.6%            | 3-5%                |
| INT2      | 2              | 16                     | 93.75%           | 5-10%               |
| INT1      | 1              | 32                     | 96.875%          | 15-20%              |

## Implementation Techniques

### 3-bit Quantization

3-bit quantization represents weights with 8 distinct values (-3, -2, -1, 0, 1, 2, 3) per weight. Unlike powers-of-two bit widths (8, 4, 2, 1), 3-bit quantization requires specialized packing and unpacking:

```typescript
// 3-bit quantization packing (example implementation)
function pack3BitValues(values: number[]): Uint32Array {
  const packedSize = Math.ceil(values.length * 3 / 32);
  const packed = new Uint32Array(packedSize);
  
  // Pack 10 complete 3-bit values per 32-bit word (10 * 3 = 30 bits)
  // Remaining 2 bits contribute to the next value
  for (let i = 0; i < values.length; i++) {
    const wordIndex = Math.floor(i * 3 / 32);
    const bitOffset = (i * 3) % 32;
    const value = values[i] & 0x7; // 3-bit mask
    
    packed[wordIndex] |= value << bitOffset;
    
    // Handle values that cross word boundaries
    if (bitOffset > 29) {
      const spillBits = 32 - bitOffset;
      packed[wordIndex + 1] |= value >>> spillBits;
    }
  }
  
  return packed;
}
```

The performance impact of 3-bit operations is minimal due to optimized WGSL shaders that handle unpacking efficiently.

### 2-bit Quantization

2-bit quantization represents weights with 4 distinct values (-1, 0, 0.5, 1) or (-1, -0.33, 0.33, 1), enabling 93.75% memory reduction with acceptable accuracy loss for many models:

```typescript
// Example 2-bit quantization with optimal value distribution
function quantize2Bit(values: Float32Array, optimization: '4values' | 'ternary'): {
  quantizedValues: Uint32Array,
  scale: Float32Array
} {
  // Choose optimal value distribution based on weight distribution
  const valueSet = optimization === '4values' 
    ? [-1, 0, 0.5, 1]  // Standard 4-value distribution
    : [-1, -0.33, 0.33, 1]; // Improved distribution for some models
  
  // Find scale factor for entire tensor (or per-channel)
  const scale = calculateOptimalScale(values, valueSet);
  
  // Map each value to closest quantized value
  const indices = values.map(v => {
    const scaled = v / scale;
    return findClosestValueIndex(scaled, valueSet);
  });
  
  // Pack 16 2-bit values per 32-bit word
  const packed = packIndices(indices, 2);
  
  return {
    quantizedValues: packed,
    scale: new Float32Array([scale])
  };
}
```

### 1-bit Quantization (Binary)

1-bit quantization is the most extreme form, representing weights with only 2 values (-1, 1):

```typescript
// 1-bit binary quantization
function quantize1Bit(values: Float32Array): {
  quantizedValues: Uint32Array,
  scale: Float32Array
} {
  // Find optimal scale
  const scale = calculateOptimalBinaryScale(values);
  
  // 32 values per 32-bit word
  const packed = new Uint32Array(Math.ceil(values.length / 32));
  
  // Pack 32 1-bit values per 32-bit word
  for (let i = 0; i < values.length; i++) {
    const wordIndex = Math.floor(i / 32);
    const bitPosition = i % 32;
    const binary = values[i] >= 0 ? 1 : 0;
    
    packed[wordIndex] |= binary << bitPosition;
  }
  
  return {
    quantizedValues: packed,
    scale: new Float32Array([scale])
  };
}
```

## Accuracy Preservation Techniques

### Per-Channel Quantization

Per-channel quantization significantly improves accuracy with ultra-low precision:

```typescript
// Pseudocode for per-channel quantization
function quantizePerChannel(
  weights: Float32Array, 
  shape: number[], 
  bitsPerWeight: 1 | 2 | 3 | 4 | 8
): {
  quantizedValues: Uint32Array,
  scales: Float32Array,
  zeroPoints: Float32Array
} {
  const [outputChannels, ...rest] = shape;
  const valuesPerChannel = rest.reduce((a, b) => a * b, 1);
  
  const scales = new Float32Array(outputChannels);
  const zeroPoints = new Float32Array(outputChannels);
  const quantizedValues = new Uint32Array(Math.ceil(weights.length * bitsPerWeight / 32));
  
  // Quantize each channel independently
  for (let c = 0; c < outputChannels; c++) {
    const channelStart = c * valuesPerChannel;
    const channelEnd = channelStart + valuesPerChannel;
    const channelValues = weights.slice(channelStart, channelEnd);
    
    // Find optimal scale/zeroPoint for this channel
    const [scale, zeroPoint] = calculateScaleZeroPoint(channelValues, bitsPerWeight);
    scales[c] = scale;
    zeroPoints[c] = zeroPoint;
    
    // Quantize and pack this channel's values
    quantizeChannelValues(
      channelValues, 
      scale, 
      zeroPoint, 
      quantizedValues, 
      channelStart, 
      bitsPerWeight
    );
  }
  
  return { quantizedValues, scales, zeroPoints };
}
```

### Mixed Precision

Mixed precision applies different bit widths to different parts of the model:

```typescript
// Pseudocode for mixed precision quantization
function applyMixedPrecision(model: Model): QuantizedModel {
  return {
    // Keep embedding layer at 8-bit
    embedding: quantize(model.embedding, 8),
    
    // Use 4-bit for intermediate layers
    encoder: model.encoder.map(layer => ({
      attention: quantize(layer.attention, 4),
      feedForward: quantize(layer.feedForward, 4)
    })),
    
    // Use 8-bit for final layer
    outputLayer: quantize(model.outputLayer, 8)
  };
}
```

### Distribution-Aware Quantization

Optimizes value distribution based on weight histograms:

```typescript
// Pseudocode for distribution-aware 2-bit quantization
function distributeOptimalValues(weights: Float32Array): number[] {
  // Analyze histogram of weights
  const histogram = calculateHistogram(weights, 100);
  
  // For 2-bit, find optimal 4 values that minimize quantization error
  const optimalValues = findOptimalValueSet(histogram, 4);
  
  return optimalValues;
}
```

## Hardware-Specific Implementation

WebGPU shaders implement efficient unpacking operations for each precision level:

### 2-bit Shader Example

```wgsl
// Simplified WGSL shader for 2-bit matrix multiplication
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 2-bit unpacking logic for WebGPU
  fn unpack_2bit(packed: u32, index: u32) -> f32 {
    let bitOffset = index * 2u;
    let value = (packed >> bitOffset) & 0x3u;
    
    // Map from 2-bit value to actual weight value
    let valueMap = array<f32, 4>(-1.0, 0.0, 0.5, 1.0);
    return valueMap[value];
  }
  
  // Matrix multiplication with 2-bit weights
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k += 16u) {
    // Each 32-bit word contains 16 2-bit weights
    let packedWord = packedWeights[weight_offset + k / 16u];
    
    for (var j: u32 = 0u; j < 16u; j++) {
      if (k + j < K) {
        let activationValue = activations[act_offset + k + j];
        let weightValue = unpack_2bit(packedWord, j) * scale[channel_idx];
        sum += activationValue * weightValue;
      }
    }
  }
  
  // Store result
  output[out_idx] = sum;
}
```

### Browser-Specific Optimizations

The framework automatically applies browser-specific optimizations for ultra-low precision:

- **Firefox**: Uses smaller workgroups (128 threads) with increased occupancy
- **Chrome**: Uses medium workgroups (256 threads) with tiling optimizations
- **Safari**: Uses larger workgroups (512 threads) with Apple-specific optimizations
- **Edge**: Integrates with WebNN when available for additional acceleration

## Usage Example

```typescript
// Configure model with ultra-low precision
const model = createBertModel(hardware, {
  modelId: 'bert-base-uncased',
  quantization: {
    enabled: true,
    bitsPerWeight: 2,  // Ultra-low 2-bit precision
    perChannelQuantization: true,
    symmetricQuantization: true,
    distributeOptimalValues: true,
    // Mixed precision configuration
    mixedPrecision: {
      firstLayerBits: 8,  // Keep first layer higher precision
      lastLayerBits: 8,   // Keep last layer higher precision
      embeddingBits: 4    // Use 4-bit for embeddings
    }
  }
});

// Initialize and use the model
await model.initialize();
const result = await model.process(input);
```

## Performance Benchmarks

| Model | Precision | Memory Usage | Accuracy | Inference Time |
|-------|-----------|--------------|----------|----------------|
| BERT-base | FP32 | 440 MB | 100% | 145 ms |
| BERT-base | INT8 | 110 MB | 99.5% | 125 ms |
| BERT-base | INT4 | 55 MB | 98% | 135 ms |
| BERT-base | INT3 | 41 MB | 96% | 138 ms |
| BERT-base | INT2 | 28 MB | 92% | 140 ms |
| ViT-base | FP32 | 344 MB | 100% | 120 ms |
| ViT-base | INT4 | 43 MB | 99% | 105 ms |
| ViT-base | INT2 | 22 MB | 94% | 110 ms |

*Benchmarks measured on RTX 3080 GPU through Chrome WebGPU implementation

## Best Practices

1. **Start with 4-bit precision** and only go lower if needed
2. **Always use per-channel quantization** for ultra-low precision
3. **Consider mixed precision** approaches for critical model parts
4. **Test accuracy** with representative inputs
5. **Try both value distributions** for 2-bit weights to find best fit
6. **Monitor browser compatibility** as WebGPU implementations evolve
7. **Benchmark both accuracy and performance** to find optimal configuration

## Conclusion

Ultra-low precision quantization enables running larger models in memory-constrained environments with acceptable accuracy tradeoffs. The IPFS Accelerate JavaScript SDK provides a complete implementation with browser-specific optimizations for maximum efficiency.