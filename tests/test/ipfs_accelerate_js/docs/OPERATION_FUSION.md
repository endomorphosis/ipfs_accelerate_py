# Operation Fusion

This guide explains how to use operation fusion in IPFS Accelerate JS to improve performance by combining multiple tensor operations into single compute shader executions.

## What is Operation Fusion?

Operation fusion is a performance optimization technique that combines multiple consecutive operations (such as matrix multiplication followed by activation) into a single compute shader. This eliminates intermediate memory transfers and buffer allocations, reducing both memory usage and computation time.

## Benefits of Operation Fusion

- **Reduced Memory Usage**: Eliminates intermediate tensors between operations
- **Faster Execution**: Fewer kernel launches and memory transfers
- **Better Cache Utilization**: Improved data locality and cache coherence
- **Lower Latency**: Fewer round-trips between CPU and GPU

## Supported Fusion Patterns

IPFS Accelerate JS supports several fusion patterns:

| Pattern | Description | Operations | Performance Gain |
|---------|-------------|------------|-----------------|
| **LinearActivation** | Matrix multiplication + activation function | matmul + relu/sigmoid/tanh/gelu | 20-30% |
| **ElementWiseChain** | Chain of element-wise operations | add/mul/sub/div chains | 40-60% |
| **BinaryUnary** | Binary operation + unary operation | add/mul/div + activation | 30-40% |
| **AttentionPattern** | Self-attention mechanism | matmul + scale + softmax + matmul | 30-50% |
| **QuantizedMatmul** | Matrix multiplication with quantized weights | 4-bit/2-bit quantized matmul | 20-40% |
| **QuantizedMatmulActivation** | Quantized matmul + activation | 4-bit matmul + activation | 30-50% |
| **MatrixChain** | Multiple matrix multiplications | matmul + matmul | 20-30% |
| **NormActivation** | Layer normalization + activation | layernorm + relu/gelu | 15-25% |

## Using Operation Fusion

### Basic Configuration

```typescript
import { createModel, FusionConfig } from 'ipfs-accelerate-js';

// Configure operation fusion
const fusionConfig: FusionConfig = {
  enabled: true,
  patterns: [
    'linear_activation',
    'elementwise_chain',
    'attention_pattern'
  ],
  useAutoFusion: true,
  maxFusionLength: 10
};

// Create model with fusion config
const model = createModel({
  modelType: 'bert',
  modelId: 'bert-base-uncased',
  operationFusion: fusionConfig
});

// Use the model normally - fusion happens automatically
const result = await model.process(input);
```

### Advanced Configuration

```typescript
// Advanced fusion configuration
const advancedFusionConfig: FusionConfig = {
  enabled: true,
  patterns: [
    'linear_activation',
    'elementwise_chain',
    'binary_unary',
    'attention_pattern',
    'quantized_matmul_activation'
  ],
  useAutoFusion: true,
  maxFusionLength: 15,
  useQuantizedWeights: true,
  bitsPerWeight: 4,
  useBrowserOptimizations: true,
  customPatterns: {
    'custom_pattern_1': ['reshape', 'matmul', 'add', 'relu'],
    'custom_pattern_2': ['add', 'mul', 'sigmoid']
  }
};
```

## Quantized Operation Fusion

For maximum performance and memory efficiency, combine operation fusion with quantization:

```typescript
// Configure quantized operation fusion
const config = {
  operationFusion: {
    enabled: true,
    patterns: [
      'quantized_matmul',
      'quantized_matmul_activation',
      'attention_pattern'
    ],
    useQuantizedWeights: true,
    bitsPerWeight: 4,
    useBrowserOptimizations: true
  }
};

// Create model with quantized operation fusion
const model = createModel({
  modelType: 'vit',
  modelId: 'google/vit-base-patch16-224',
  ...config
});
```

## Browser-Specific Optimizations

Operation fusion can be optimized for specific browsers:

```typescript
// Configure browser-specific optimizations
const browserOptimizedConfig = {
  operationFusion: {
    enabled: true,
    useBrowserOptimizations: true,
    // The SDK will automatically detect the browser type
    // and apply appropriate optimizations
    patterns: [
      'linear_activation',
      'attention_pattern'
    ]
  }
};
```

## Manual Fusion for Advanced Use Cases

For more control, you can manually create fused operations:

```typescript
import { WebGPUBackend, WebGPUOperationFusion, FusionPattern } from 'ipfs-accelerate-js';

// Initialize backend
const backend = new WebGPUBackend();
await backend.initialize();

// Configure fusion
const fusion = new WebGPUOperationFusion(backend, {
  useQuantizedWeights: true,
  bitsPerWeight: 4,
  useBrowserOptimizations: true
});

// Execute fused operations directly
const result = await fusion.executeFusedOperations(
  [inputTensor, weightTensor],
  ['matmul', 'relu']
);
```

## Performance Tips

1. **Prioritize Memory-Bound Operations**: Focus on fusion patterns that eliminate large intermediate tensors
2. **Combine with Quantization**: For maximum performance, combine operation fusion with weight quantization
3. **Layer Selection**: Focus on compute-intensive layers like attention mechanisms and linear transformations
4. **Browser Testing**: Different browsers have different WebGPU implementations; test on all target browsers
5. **Warm-up Runs**: First execution includes compilation overhead; warm up with a few inference runs

## Example: Attention Mechanism Optimization

The attention mechanism in transformer models is an excellent candidate for operation fusion:

```typescript
// Configure attention fusion for transformer models
const transformerConfig = {
  operationFusion: {
    enabled: true,
    patterns: ['attention_pattern'],
    useBrowserOptimizations: true
  }
};

// Create BERT model with attention fusion
const bertModel = createBertModel(hardware, {
  modelId: 'bert-base-uncased',
  ...transformerConfig
});

// The attention computation (Q*K^T/sqrt(dk) -> softmax -> *V)
// will be fused into a single compute shader
```

## Debugging Fusion Patterns

To verify which fusion patterns are being used:

```typescript
// Enable fusion debugging
const debugConfig = {
  operationFusion: {
    enabled: true,
    patterns: ['linear_activation', 'attention_pattern'],
    debug: true  // Logs fusion patterns being applied
  }
};

// Output will show which patterns were detected and fused
```