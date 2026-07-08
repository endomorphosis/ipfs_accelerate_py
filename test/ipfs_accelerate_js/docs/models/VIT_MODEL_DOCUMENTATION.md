# Vision Transformer (ViT) Model Documentation

The ViT implementation in the IPFS Accelerate JS SDK provides hardware-accelerated image classification capabilities directly in the browser using WebGPU and WebNN backends. 

## Overview

The Vision Transformer (ViT) architecture treats images as sequences of patches, similar to how transformers process sequences of words in NLP. This approach has shown impressive results on image classification tasks, rivaling or exceeding traditional convolutional neural networks.

Key features of our implementation:

- **Hardware Acceleration**: Utilizes WebGPU and WebNN for GPU acceleration
- **Browser Optimization**: Implements browser-specific optimizations for best performance
- **Memory Efficiency**: Careful tensor management with explicit cleanup
- **Cross-Model Sharing**: Enables sharing embeddings between models
- **Standard ViT Variants**: Support for common ViT variants (base, large)

## Usage

### Basic Example

```typescript
import { createVitModel, ViTConfig, ViTInput } from 'ipfs-accelerate-js';
import { createWebGPUBackend } from 'ipfs-accelerate-js';

async function classifyImage(imageData: Uint8Array, width: number, height: number) {
  // Create WebGPU backend
  const hardware = await createWebGPUBackend();
  await hardware.initialize();
  
  // Create ViT model
  const config: Partial<ViTConfig> = {
    modelId: 'google/vit-base-patch16-224',
    useOptimizedOps: true
  };
  
  const model = createVitModel(hardware, config);
  await model.initialize();
  
  // Prepare input
  const input: ViTInput = {
    imageData,
    width,
    height
  };
  
  // Run inference
  const result = await model.process(input);
  
  // Use the classification results
  console.log(`Predicted class: ${result.classId}`);
  console.log(`Probability: ${result.probabilities[result.classId]}`);
  
  // Clean up resources
  await model.dispose();
}
```

### Preprocessing Images

The ViT model expects images to be preprocessed to match its input requirements:

1. Resize the image to the expected input dimensions (typically 224×224 pixels)
2. Convert the pixel values to the expected range (0-255 or 0-1)
3. Arrange the image data in the expected format (RGB/BGR, channel-first/channel-last)

```typescript
// Example of preprocessing an image from an HTML image element
function preprocessImage(image: HTMLImageElement): Uint8Array {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d')!;
  
  // Draw and resize image
  ctx.drawImage(image, 0, 0, 224, 224);
  
  // Get image data (RGBA format)
  const imageData = ctx.getImageData(0, 0, 224, 224);
  
  // Convert to RGB format (remove alpha channel)
  const rgbData = new Uint8Array(224 * 224 * 3);
  for (let i = 0; i < imageData.data.length / 4; i++) {
    rgbData[i * 3] = imageData.data[i * 4];     // R
    rgbData[i * 3 + 1] = imageData.data[i * 4 + 1]; // G
    rgbData[i * 3 + 2] = imageData.data[i * 4 + 2]; // B
  }
  
  return rgbData;
}
```

## Configuration

The ViT model can be configured with the following options:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `modelId` | `string` | Model identifier | `'google/vit-base-patch16-224'` |
| `imageSize` | `number` | Input image size (width/height) | `224` |
| `patchSize` | `number` | Patch size for tokenization | `16` |
| `hiddenSize` | `number` | Hidden dimension size | `768` |
| `numLayers` | `number` | Number of Transformer layers | `12` |
| `numHeads` | `number` | Number of attention heads | `12` |
| `intermediateSize` | `number` | Intermediate feed-forward size | `3072` |
| `layerNormEps` | `number` | Layer normalization epsilon | `1e-12` |
| `numClasses` | `number` | Number of output classes | `1000` |
| `backendPreference` | `TensorBackendType[]` | Preferred backends in order | `['webgpu', 'webnn', 'cpu']` |
| `useOptimizedOps` | `boolean` | Whether to use optimized operations | `true` |
| `channels` | `number` | Number of channels in input image | `3` |

### Supported ViT Models

| Model ID | Patch Size | Hidden Size | Layers | Heads | Parameters |
|----------|------------|-------------|--------|-------|------------|
| `google/vit-base-patch16-224` | 16 | 768 | 12 | 12 | 86M |
| `google/vit-large-patch16-224` | 16 | 1024 | 24 | 16 | 307M |
| `google/vit-huge-patch14-224` | 14 | 1280 | 32 | 16 | 632M |

## Advanced Features

### Cross-Model Tensor Sharing

The ViT model supports sharing its internal representations with other models through the `SharedTensor` system:

```typescript
// Extract vision embeddings for use with other models
const vitModel = createVitModel(hardware, config);
await vitModel.initialize();

// Process image and create shared tensor
const result = await vitModel.process(input);
const sharedTensor = vitModel.getSharedTensor('vision_embedding');

// Use shared tensor with another model (e.g., CLIP)
const clipModel = createClipModel(hardware, {
  useSharedEmbeddings: true
});
await clipModel.initialize();

// Pass the shared tensor to CLIP
const clipResult = await clipModel.processWithSharedEmbedding(
  textInput, 
  sharedTensor
);
```

### Browser-Specific Optimizations

The ViT implementation includes browser-specific optimizations for WebGPU:

```typescript
// Create a ViT model with Chrome-specific optimizations
const vitModel = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224',
  useOptimizedOps: true,
  browserOptimizations: {
    chrome: {
      workgroupSize: 256,
      useReducedPrecision: true,
      usePipelineCache: true
    }
  }
});
```

### Memory Management

The ViT model includes careful memory management to avoid memory leaks:

```typescript
// Create and use the model within a try-finally block to ensure cleanup
let vitModel = null;
try {
  vitModel = createVitModel(hardware, config);
  await vitModel.initialize();
  
  // Use the model...
  const result = await vitModel.process(input);
  
  // Process the results...
} finally {
  // Always clean up resources
  if (vitModel) {
    await vitModel.dispose();
  }
}
```

## Performance Considerations

### Hardware Acceleration

The ViT model performs best with hardware acceleration:

- **WebGPU**: Available in Chrome 113+, Edge 113+, and Firefox 118+
- **WebNN**: Available in Edge with experimental flags

### Performance Comparison

| Backend | Inference Time (ms) | Memory Usage |
|---------|---------------------|--------------|
| WebGPU  | 25-40ms             | Medium       |
| WebNN   | 20-35ms             | Low          |
| CPU     | 150-300ms           | High         |

### Optimizing Performance

1. **Use appropriate model size**: Smaller models (base vs large) are faster
2. **Enable browser-specific optimizations**: Set `useOptimizedOps: true`
3. **Reuse models**: Initialize once, process multiple images
4. **Share tensors**: Use the cross-model sharing feature when applicable

## Browser Compatibility

| Browser | WebGPU | WebNN | CPU Fallback |
|---------|--------|-------|--------------|
| Chrome 113+ | ✅ | ❌ | ✅ |
| Edge 113+ | ✅ | ✅ | ✅ |
| Firefox 118+ | ✅ | ❌ | ✅ |
| Safari 17+ | ✅ | ❌ | ✅ |

## Error Handling

The ViT model implementation includes comprehensive error handling:

```typescript
try {
  const result = await vitModel.process(input);
  // Handle successful result
} catch (error) {
  if (error instanceof WebGPUNotSupportedError) {
    // Handle WebGPU not supported
    console.log("WebGPU not supported, trying fallback...");
  } else if (error instanceof ModelInitializationError) {
    // Handle model initialization error
    console.error("Failed to initialize model:", error.message);
  } else if (error instanceof InferenceError) {
    // Handle inference error
    console.error("Inference failed:", error.message);
  } else {
    // Handle other errors
    console.error("Unexpected error:", error);
  }
}
```

## Example Applications

- **Image Classification**: Classify images into 1000 ImageNet categories
- **Visual Search**: Extract features for similarity-based image retrieval
- **Multimodal Systems**: Combine with text models for image-text tasks
- **Transfer Learning**: Use as a feature extractor for custom tasks

## Related Components

- **Tensor Operations**: Core tensor operations for matrix calculations
- **WebGPU Backend**: Hardware-accelerated tensor operations via WebGPU
- **WebNN Backend**: Neural network acceleration via WebNN
- **SharedTensor**: Cross-model tensor sharing mechanism
- **Hardware Detection**: Browser and hardware capability detection

## Further Resources

- [ViT Architecture Paper](https://arxiv.org/abs/2010.11929)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WebNN API](https://www.w3.org/TR/webnn/)
- [IPFS Accelerate JS API Reference](../api/README.md)