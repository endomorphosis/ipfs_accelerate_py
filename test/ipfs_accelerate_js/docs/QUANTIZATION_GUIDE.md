# Quantization Guide

This guide explains how to use quantization techniques in IPFS Accelerate JS to reduce memory usage and improve performance of AI models in web browsers.

## What is Quantization?

Quantization is a technique that reduces the precision of the numbers used to represent model weights and activations. By converting 32-bit floating point numbers to lower precision formats (8-bit, 4-bit, 3-bit, or even 2-bit), we can significantly reduce memory usage and speed up memory transfers, with minimal impact on model accuracy.

## Benefits of Quantization

- **Reduced Memory Usage**: Up to 16x smaller models with 2-bit quantization
- **Faster Loading**: Less data to transfer from storage to GPU
- **Better Performance**: Reduced memory bandwidth requirements
- **Support for Larger Models**: Run models that would otherwise be too large for browser memory

## Quantization Types

IPFS Accelerate JS supports several quantization methods:

### By Precision Level

| Precision | Memory Reduction | Impact on Accuracy | Use Case |
|-----------|------------------|-------------------|----------|
| 8-bit     | 4x smaller       | Negligible        | General purpose with highest accuracy |
| 4-bit     | 8x smaller       | Very low          | Most models, balanced approach |
| 3-bit     | ~11x smaller     | Moderate          | Memory-constrained devices |
| 2-bit     | 16x smaller      | Higher            | Extreme memory constraints |

### By Technique

- **Symmetric Quantization**: Uses a zero-centered range, better for weights with balanced distributions
- **Asymmetric Quantization**: Uses full range, better for activations with unbalanced distributions
- **Per-Tensor Quantization**: Uses single scale/zero-point for entire tensor, simplest approach
- **Per-Channel Quantization**: Uses scale/zero-point for each channel, better accuracy but more metadata

## Using Quantization

### Basic Configuration

```typescript
import { createModel, QuantizationConfig } from 'ipfs-accelerate-js';

// Configure 4-bit quantization
const quantConfig: QuantizationConfig = {
  enabled: true,
  bitsPerWeight: 4,           // Use 4-bit precision for weights
  bitsPerActivation: 8,       // Use 8-bit for activations
  includeFirstLayer: false,   // Keep first layer in full precision
  includeLastLayer: false     // Keep last layer in full precision
};

// Create model with quantization config
const model = createModel({
  modelType: 'vit',
  modelId: 'google/vit-base-patch16-224',
  quantization: quantConfig
});

// Use the model normally
const result = await model.process(input);
```

### Advanced Configuration

```typescript
// Advanced quantization configuration
const advancedQuantConfig: QuantizationConfig = {
  enabled: true,
  bitsPerWeight: 3,                // 3-bit precision
  symmetricQuantization: true,     // Use symmetric quantization for weights
  perChannelQuantization: true,    // Per-channel quantization for better accuracy
  customScaleFactors: {            // Custom scale factors for specific layers
    'encoder.layer.0': 0.1,
    'encoder.layer.11': 0.1
  },
  excludeLayers: ['embeddings'],   // Layers to exclude from quantization
  calibrationSamples: 100,         // Number of samples to use for calibration
  dequantizeOnGPU: true            // Perform dequantization on GPU
};
```

## Ultra-Low Precision (2-bit/3-bit)

For extreme memory constraints, use 2-bit or 3-bit quantization:

```typescript
const ultraLowPrecisionConfig: QuantizationConfig = {
  enabled: true,
  bitsPerWeight: 2,                // 2-bit precision (16x smaller than FP32)
  symmetricQuantization: true,     // Required for ultra-low precision
  perChannelQuantization: true,    // Strongly recommended for accuracy
  excludeLayers: [                 // Keep critical layers in higher precision
    'embeddings',
    'pooler',
    'classifier'
  ],
  optimizeForAccuracy: true        // Use optimized techniques for accuracy preservation
};
```

## Dynamic Quantization

For optimizing memory usage depending on device capabilities:

```typescript
import { getDeviceMemoryInfo, determineOptimalBitWidth } from 'ipfs-accelerate-js';

// Dynamically determine best bit width based on device memory
const memoryInfo = await getDeviceMemoryInfo();
const optimalBitWidth = determineOptimalBitWidth(
  memoryInfo,
  'vit-base-patch16-224',  // Model ID
  { minAccuracy: 0.95 }    // Minimal acceptable accuracy
);

// Configure quantization dynamically
const dynamicConfig: QuantizationConfig = {
  enabled: true,
  bitsPerWeight: optimalBitWidth,
  symmetricQuantization: optimalBitWidth <= 3, // Use symmetric for ultra-low precision
  perChannelQuantization: true
};
```

## Integration with Operation Fusion

Quantization works seamlessly with operation fusion for additional performance:

```typescript
// Configure quantization with operation fusion
const config = {
  quantization: {
    enabled: true,
    bitsPerWeight: 4
  },
  operationFusion: {
    enabled: true,
    useQuantizedOps: true,   // Use specialized quantized kernels
    patterns: ['linear_activation', 'attention_pattern']
  }
};

// Create model with both optimizations
const model = createModel({
  modelType: 'bert',
  modelId: 'bert-base-uncased',
  ...config
});
```

## Memory Savings

Here's the approximate memory savings for different model sizes with quantization:

| Model Size | FP32 (32-bit) | INT8 (8-bit) | INT4 (4-bit) | INT2 (2-bit) |
|------------|---------------|--------------|--------------|--------------|
| ViT Base   | 329 MB        | 82 MB        | 41 MB        | 21 MB        |
| BERT Base  | 436 MB        | 109 MB       | 55 MB        | 27 MB        |
| Whisper Tiny | 152 MB      | 38 MB        | 19 MB        | 10 MB        |

## Performance Considerations

- **Memory Bandwidth**: Quantization reduces memory bandwidth requirements, improving performance for memory-bound operations
- **Computation**: Dequantization adds some computational overhead, but is typically outweighed by bandwidth savings
- **Browser Compatibility**: All modern browsers supporting WebGPU can take advantage of quantization
- **Mobile Devices**: Particularly beneficial on mobile devices with limited memory and bandwidth

## Best Practices

1. **Start Conservative**: Begin with 8-bit or 4-bit quantization before trying ultra-low precision
2. **Evaluate Accuracy**: Always test model accuracy with your quantization settings
3. **Layer Selectivity**: Consider excluding critical layers (first/last/embedding) from quantization
4. **Per-Channel for Lower Bits**: Always use per-channel quantization for 2-bit and 3-bit precision
5. **Combine with Fusion**: Pair quantization with operation fusion for best performance