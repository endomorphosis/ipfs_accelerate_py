# WebGPU Tensor Sharing Implementation Guide

This guide provides detailed information about the WebGPU Tensor Sharing system for the IPFS Accelerate JavaScript SDK, explaining its architecture, key components, and implementation details.

## Overview

The WebGPU Tensor Sharing system integrates Cross-Model Tensor Sharing with WebGPU compute capabilities to provide hardware-accelerated tensor operations while maintaining efficient memory sharing between models. It enables efficient GPU-accelerated machine learning in web browsers with optimizations for different browsers and hardware configurations.

## Architecture

The system consists of these primary components:

1. **WebGPUTensorSharing**: Core class that integrates with the TensorSharingIntegration and WebGPUBackend
2. **TensorSharingIntegration**: Manages shared tensors, reference counting, and memory management
3. **WebGPUBackend**: Provides WebGPU hardware acceleration for tensor operations
4. **Browser-Specific Optimizations**: Detects and optimizes for different browsers and hardware
5. **SharedTensors**: Memory-efficient tensor representations that can be shared between models

### System Diagram

```
┌─────────────────────┐     ┌─────────────────┐     ┌───────────────┐
│  Model Application  │────▶│  TensorSharing  │◀───▶│ StorageManager│
└──────────┬──────────┘     └────────┬────────┘     └───────────────┘
           │                         │                       
           ▼                         ▼                       
┌─────────────────────┐     ┌────────────────┐      ┌───────────────┐
│ WebGPUTensorSharing │◀───▶│ WebGPUBackend  │◀────▶│ Browser-Specific│
└──────────┬──────────┘     └────────┬───────┘      │ Optimizations  │
           │                         │              └───────────────┘
           ▼                         ▼              
┌─────────────────────┐     ┌────────────────┐     
│    GPU Buffers      │◀───▶│  WebGPU Device │     
└─────────────────────┘     └────────────────┘     
```

## Core Components

### WebGPUTensorSharing Class

The central class that orchestrates the integration between tensor sharing and WebGPU:

```typescript
class WebGPUTensorSharing {
  constructor(tensorSharing: TensorSharingIntegration, options?: WebGPUTensorSharingOptions);
  async initialize(): Promise<boolean>;
  async matmul(tensorA: string, modelA: string, tensorB: string, modelB: string, 
              outputTensorName: string, modelName: string, 
              transposeA?: boolean, transposeB?: boolean): Promise<string>;
  async elementwise(inputTensorName: string, inputModelName: string, 
                   outputTensorName: string, outputModelName: string, 
                   operation?: 'relu' | 'sigmoid' | 'tanh'): Promise<string>;
  async softmax(inputTensorName: string, inputModelName: string, 
               outputTensorName: string, outputModelName: string, 
               axis?: number): Promise<string>;
  async quantize(inputTensorName: string, inputModelName: string, 
                outputTensorName: string, outputModelName: string): Promise<{tensorName: string, scaleTensorName: string}>;
  async dequantize(inputTensorName: string, inputModelName: string, 
                  scaleTensorName: string, scaleModelName: string, 
                  outputTensorName: string, outputModelName: string): Promise<string>;
  async createTensorView(sourceTensorName: string, sourceModelName: string, 
                        viewTensorName: string, targetModelName: string, 
                        offset?: number, size?: number): Promise<string>;
  async shareTensorBetweenModels(tensorName: string, sourceModelName: string, 
                                targetModelNames: string[]): Promise<void>;
  async synchronizeTensor(tensorName: string, modelName: string, 
                         direction?: 'cpu-to-gpu' | 'gpu-to-cpu' | 'both'): Promise<void>;
  async createCustomShader(shaderName: string, shaderCode: string, 
                          inputTypes?: Record<string, 'float32' | 'int32' | 'uint8'>, 
                          outputTypes?: Record<string, 'float32' | 'int32' | 'uint8'>): Promise<void>;
  async executeCustomShader(shaderName: string, 
                           inputTensors: Record<string, {tensorName: string, modelName: string}>, 
                           outputTensorNames: Record<string, {tensorName: string, shape: number[], dataType: 'float32' | 'int32' | 'uint8'}>, 
                           modelName: string, 
                           workgroupSize?: [number, number, number], 
                           workgroupCount?: [number, number, number]): Promise<Record<string, string>>;
  getGPUMemoryUsage(): number;
  optimizeMemoryUsage(): Promise<void>;
  dispose(): void;
}
```

### Configuration Options

The WebGPUTensorSharing class accepts several configuration options:

```typescript
interface WebGPUTensorSharingOptions {
  // WebGPU backend to use, if not provided, a new one will be created
  webgpuBackend?: WebGPUBackend;
  
  // Storage manager to use for persistent storage
  storageManager?: StorageManager;
  
  // WebGPU backend options if creating a new backend
  webgpuOptions?: any;
  
  // Enable browser-specific optimizations for shaders
  browserOptimizations?: boolean;
  
  // Browser type to optimize for (auto-detected if not specified)
  browserType?: BrowserType;
  
  // Custom shader optimization settings
  shaderOptimizationSettings?: Record<string, any>;
  
  // Enable zero-copy mode where possible
  enableZeroCopy?: boolean;
  
  // Cache compute pipelines for better performance
  cachePipelines?: boolean;
  
  // Priority mode for tensor operations: 'speed', 'memory', or 'balanced'
  priorityMode?: 'speed' | 'memory' | 'balanced';
  
  // Debug mode to provide detailed logs and validations
  debug?: boolean;
}
```

## Implementation Details

### Memory Management

The WebGPU Tensor Sharing system employs sophisticated memory management:

1. **Tensor Buffer Cache**: Keeps track of GPU-resident tensors
   ```typescript
   private tensorBufferCache: Map<string, {
     buffer: GPUBuffer;
     shape: number[];
     dataType: string;
     location: SharedTensorLocation;
     lastUsed: number;
     size: number;
   }> = new Map();
   ```

2. **Pipeline Cache**: Stores compiled compute pipelines for reuse
   ```typescript
   private pipelineCache: Map<string, {
     pipeline: GPUComputePipeline;
     bindGroupLayout: GPUBindGroupLayout;
     lastUsed: number;
   }> = new Map();
   ```

3. **Shader Cache**: Stores optimized shaders for different operations
   ```typescript
   private shaderCache: Map<string, string> = new Map();
   ```

4. **Memory Optimization**: Automatically moves tensors between CPU and GPU
   ```typescript
   async optimizeMemoryUsage(): Promise<void> {
     // Identify rarely used GPU tensors
     // Move them to CPU if needed
     // Clean up unused tensors
   }
   ```

### Browser-Specific Optimizations

The system automatically detects browser type and capabilities:

```typescript
// During initialization
if (this.options.browserOptimizations !== false) {
  const device = (this.webgpuBackend as any)['device'];
  if (device) {
    this.browserCapabilities = await getBrowserCapabilities(device);
    await this.precompileOptimizedShaders();
  }
}
```

Optimizations include:

1. **Browser-Specific Shader Generation**: Creates optimized WGSL code for each browser
2. **Adaptive Workgroup Sizes**: Chooses optimal workgroup sizes based on browser and operation
3. **Hardware-Specific Tuning**: Adapts to GPU vendor and architecture

### Zero-Copy Tensor Views

The system supports zero-copy tensor views:

```typescript
async createTensorView(
  sourceTensorName: string,
  sourceModelName: string,
  viewTensorName: string,
  targetModelName: string,
  offset: number = 0,
  size?: number
): Promise<string> {
  // Create view in tensor sharing system
  // Create GPU buffer view if source tensor is GPU-resident
  // Return the view tensor name
}
```

This enables efficient memory use by avoiding redundant copies of tensor data.

## Using Custom Shaders

The WebGPU Tensor Sharing system supports custom compute shaders:

```typescript
// Define custom shader
const dotProductShader = `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> result: array<f32>;
  
  @compute @workgroup_size(1)
  fn main() {
    var dot_product = 0.0;
    for (var i = 0u; i < arrayLength(&a); i = i + 1u) {
      dot_product = dot_product + a[i] * b[i];
    }
    result[0] = dot_product;
  }
`;

// Register the shader
await webgpuTensorSharing.createCustomShader('dot_product', dotProductShader);

// Execute the shader
await webgpuTensorSharing.executeCustomShader(
  'dot_product',
  { 
    a: { tensorName: 'text_embedding', modelName: 'clip-model' },
    b: { tensorName: 'vision_embedding', modelName: 'clip-model' }
  },
  { 
    result: { tensorName: 'similarity_score', shape: [1], dataType: 'float32' }
  },
  'clip-model'
);
```

## Tensor Operation Example

Here's an example of matrix multiplication with browser-optimized shaders:

```typescript
// With tensor A of shape [M, K] and tensor B of shape [K, N]
const outputTensorName = await webgpuTensorSharing.matmul(
  'tensorA', 'modelA',  // First tensor and its owner model
  'tensorB', 'modelB',  // Second tensor and its owner model
  'output', 'modelC',   // Output tensor name and owner model
  false, false          // No transposes
);

// The matrix multiplication will use browser-optimized shaders
// with appropriate workgroup sizes for the current browser
```

The resulting tensor is registered with the tensor sharing system, allowing it to be used by other models.

## Advanced Usage Patterns

### Multimodal Workflow

A common use case is processing multiple modalities (text and vision) with different models:

```typescript
// Process text with BERT
const bertHiddenStates = await bertModel.process(textInput);
await tensorSharing.registerSharedTensor(
  'bert_hidden_states',
  bertHiddenStates.shape,
  bertHiddenStates.data,
  'float32',
  'bert-model',
  ['bert-model']
);

// Process image with ViT
const vitEmbeddings = await vitModel.process(imageInput);
await tensorSharing.registerSharedTensor(
  'vit_embeddings',
  vitEmbeddings.shape,
  vitEmbeddings.data,
  'float32',
  'vit-model',
  ['vit-model']
);

// Share with CLIP model
await webgpuTensorSharing.shareTensorBetweenModels(
  'bert_hidden_states', 'bert-model', ['clip-model']
);
await webgpuTensorSharing.shareTensorBetweenModels(
  'vit_embeddings', 'vit-model', ['clip-model']
);

// Project embeddings to common space
await webgpuTensorSharing.matmul(
  'bert_hidden_states', 'clip-model',
  'text_projection_matrix', 'clip-model',
  'text_embedding', 'clip-model'
);

await webgpuTensorSharing.matmul(
  'vit_embeddings', 'clip-model',
  'vision_projection_matrix', 'clip-model',
  'vision_embedding', 'clip-model'
);

// Compute similarity
await webgpuTensorSharing.executeCustomShader(
  'cosine_similarity',
  {
    a: { tensorName: 'text_embedding', modelName: 'clip-model' },
    b: { tensorName: 'vision_embedding', modelName: 'clip-model' }
  },
  {
    result: { tensorName: 'similarity_score', shape: [1], dataType: 'float32' }
  },
  'clip-model'
);
```

### Memory Optimization

Optimize memory usage during long-running applications:

```typescript
// Periodically optimize memory usage
setInterval(async () => {
  await webgpuTensorSharing.optimizeMemoryUsage();
  
  // Log memory usage
  const gpuMemory = webgpuTensorSharing.getGPUMemoryUsage();
  console.log(`GPU memory usage: ${gpuMemory / (1024 * 1024)} MB`);
}, 30000); // Every 30 seconds
```

## Error Handling

The WebGPU Tensor Sharing system includes comprehensive error handling:

```typescript
try {
  // Try to execute a tensor operation
  await webgpuTensorSharing.matmul(
    'tensorA', 'modelA',
    'tensorB', 'modelB',
    'output', 'modelC'
  );
} catch (error) {
  if (error.message.includes('WebGPU not supported')) {
    // Handle WebGPU not being available
    console.warn('WebGPU not supported, falling back to CPU implementation');
    
    // Perform fallback CPU implementation
    // ...
  } else if (error.message.includes('dimension mismatch')) {
    // Handle tensor dimension mismatch
    console.error('Matrix dimension mismatch:', error);
    
    // Attempt to reshape tensors or use different ones
    // ...
  } else {
    // Handle other errors
    console.error('Unexpected error during tensor operation:', error);
  }
}
```

## Cleanup

When you're done with the WebGPU Tensor Sharing system, clean up resources:

```typescript
// Dispose of all resources
webgpuTensorSharing.dispose();
```

This releases GPU buffers, shader modules, and other WebGPU resources.

## Debugging

Enable debug mode to get detailed information:

```typescript
const webgpuTensorSharing = new WebGPUTensorSharing(tensorSharing, {
  debug: true
});

// This will log:
// - Browser and hardware detection results
// - Shader compilation and optimization details
// - Memory usage information
// - Operation execution details
```

## Performance Considerations

For optimal performance:

1. **Reuse tensors** when possible instead of creating new ones
2. **Batch operations** rather than executing many small operations
3. **Consider quantization** for large models to reduce memory usage
4. **Use browser-specific optimizations** for maximum performance
5. **Minimize CPU-GPU synchronization** by keeping data on the GPU when possible
6. **Use tensor views** instead of copying data when only a portion is needed
7. **Periodically optimize memory** to avoid excessive GPU memory usage

## Related Documentation

For more information, see:
- [WebGPU Tensor Sharing README](WEBGPU_TENSOR_SHARING_README.md)
- [Cross-Model Tensor Sharing Guide](CROSS_MODEL_TENSOR_SHARING_GUIDE.md)
- [Browser-Specific Optimization Guide](BROWSER_OPTIMIZATION_GUIDE.md)
- [WebGPU Operation Fusion Guide](OPERATION_FUSION_GUIDE.md)