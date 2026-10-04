# Ultra-Low Precision Quantization Guide

## Overview

This guide provides detailed information about the ultra-low precision quantization techniques implemented in the Model Conversion Generator. These techniques enable significant memory reduction while maintaining acceptable model accuracy.

## Quantization Levels

The framework supports multiple precision levels:

| Precision | Bits per Value | Values per Byte | Memory Reduction | Use Case |
|-----------|---------------|-----------------|-----------------|----------|
| float32   | 32            | 0.25            | 0%              | Full precision (baseline) |
| float16   | 16            | 0.5             | 50%             | High-accuracy requirements |
| int8/8-bit| 8             | 1               | 75%             | Good accuracy-size balance |
| 4-bit     | 4             | 2               | 87.5%           | Small models, acceptable accuracy |
| 3-bit     | 3             | 2.67            | 90.6%           | Memory-constrained environments |
| 2-bit     | 2             | 4               | 93.75%          | Extreme memory optimization |

## Technical Implementation

### 4-bit Quantization

Each byte stores two 4-bit values. The implementation includes:

1. **Quantization**: Converting floating-point values to 4-bit integers
2. **Packing**: Packing two 4-bit values into a single byte
3. **Dequantization**: Converting packed 4-bit values back to floating-point

```wgsl
// Extract 4-bit values from a byte
let value1 = byte & 0x0F;
let value2 = (byte >> 4) & 0x0F;

// Pack two 4-bit values into a byte
packed_byte = u8(value1 | (value2 << 4));
```

### 3-bit Quantization

3-bit quantization is more complex since 3 doesn't divide evenly into 8 (byte size). The implementation packs 8 3-bit values into 3 bytes:

```wgsl
// Pack the 8 3-bit values into 3 bytes
// First byte: contains v1 (3 bits) + v2 (3 bits) + first 2 bits of v3
byte1 = u8(v1 | (v2 << 3) | ((v3 & 0x03) << 6));
// Second byte: contains last 1 bit of v3 + v4 (3 bits) + v5 (3 bits) + first 1 bit of v6
byte2 = u8(((v3 & 0x04) >> 2) | (v4 << 1) | (v5 << 4) | ((v6 & 0x01) << 7));
// Third byte: contains last 2 bits of v6 + v7 (3 bits) + v8 (3 bits)
byte3 = u8(((v6 & 0x06) >> 1) | (v7 << 2) | (v8 << 5));
```

### 2-bit Quantization

2-bit quantization represents each value with only 4 possible values (0-3), packing 4 values into each byte:

```wgsl
// Extract 2-bit values from a byte
let val1 = byte & 0x03;
let val2 = (byte >> 2) & 0x03;
let val3 = (byte >> 4) & 0x03;
let val4 = (byte >> 6) & 0x03;

// Pack four 2-bit values into one byte
packed_byte = u8(v1 | (v2 << 2) | (v3 << 4) | (v4 << 6));
```

## Mixed Precision Approach

For optimal accuracy-size tradeoffs, the framework uses mixed precision quantization:

### Layer-Specific Precision

Different layers use different precision levels based on their sensitivity:

```wgsl
struct MixedPrecisionConfig {
  use_2bit : array<u32, 16>,  // Layer indices that use 2-bit
  use_3bit : array<u32, 16>,  // Layer indices that use 3-bit
  use_4bit : array<u32, 16>,  // Layer indices that use 4-bit
  use_8bit : array<u32, 16>,  // Layer indices that use 8-bit
  use_fp16 : array<u32, 16>,  // Layer indices that use fp16
  num_layers : u32,           // Total number of layers
}
```

### Component-Specific Precision

Different components of the model use precision levels based on their importance:

1. **Attention Layers**: Typically need higher precision (fp16 or 8-bit)
2. **Feed-Forward Networks**: Can use lower precision (4-bit or 3-bit)
3. **Embeddings**: Often use medium precision (8-bit)
4. **Prediction Heads**: Typically require higher precision (fp16 or 8-bit)

## Calibration for Optimal Quantization

The quantization process includes:

1. **Post-Training Analysis**: Analyze weight distributions
2. **Scale Factor Determination**: Calculate optimal scale factors for each layer
3. **Zero-Point Calibration**: Determine zero-points to minimize quantization error
4. **Per-Channel Quantization**: Use different scale factors for different channels
5. **Outlier Handling**: Special handling for outlier values

## Usage Examples

### Basic Usage

```bash
# Convert ONNX model to WebGPU with 4-bit quantization
python -m refactored_model_conversion_generator convert \
    --source model.onnx \
    --source-format onnx \
    --target-format webgpu \
    --precision 4bit \
    --output model.webgpu.js
```

### Advanced Configuration

For fine-grained control, you can create a configuration file specifying precision for each layer:

```json
{
  "model_name": "bert-base-uncased",
  "mixed_precision": {
    "2bit_layers": [4, 5, 6, 7],
    "3bit_layers": [8, 9, 10, 11],
    "4bit_layers": [0, 1, 2, 3],
    "8bit_layers": [12],
    "fp16_layers": []
  }
}
```

## Performance Impact

| Precision | Memory Savings | Accuracy Impact | Inference Speed |
|-----------|---------------|-----------------|----------------|
| float16   | 50%           | <1%             | 1.2-1.5x faster|
| int8      | 75%           | 1-2%            | 1.5-2x faster  |
| 4-bit     | 87.5%         | 3-5%            | 2-3x faster    |
| 3-bit     | 90.6%         | 5-10%           | 2-3x faster    |
| 2-bit     | 93.75%        | 10-20%          | 3-4x faster    |

*Note: Actual performance varies by model architecture and hardware.*

## Limitations and Considerations

1. **Accuracy Trade-offs**: Lower precision generally means lower accuracy
2. **Model-Specific Behavior**: Different models respond differently to quantization
3. **Hardware Support**: Some precision levels work better on specific hardware
4. **Activation Quantization**: Currently focused on weight quantization; activation quantization is future work
5. **Fine-tuning**: For best results, consider fine-tuning after quantization

## References

This implementation draws from research in ultra-low precision neural networks:

1. The WEBGPU_4BIT_INFERENCE_README.md documentation
2. The WEBGPU_WEBNN_QUANTIZATION_GUIDE.md guide
3. The WEBNN_WEBGPU_QUANTIZATION_MARCH2025_UPDATE.md update
4. Recent research on 3-bit and 2-bit neural network quantization