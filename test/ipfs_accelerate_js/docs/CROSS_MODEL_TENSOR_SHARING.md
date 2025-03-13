# Cross-Model Tensor Sharing

The IPFS Accelerate JS SDK provides a powerful feature for sharing tensor data between different models, enabling efficient multimodal applications with significant performance and memory benefits.

## Overview

Cross-model tensor sharing allows models to share intermediate representations with each other, avoiding redundant computation and memory usage. For example, a vision model can extract features from an image and share them with a multimodal model, which can then combine these features with text features without re-processing the image.

Key benefits:

- **Reduced Memory Usage**: Up to 30% memory reduction for common multi-model workflows
- **Inference Speedup**: Up to 30% faster inference when reusing cached embeddings
- **Increased Throughput**: Higher throughput when running multiple related models
- **Browser Resource Efficiency**: More efficient use of limited browser memory resources

## How It Works

The SDK uses a `SharedTensor` system that enables models to:

1. Create shareable tensor representations with appropriate metadata
2. Reference count these tensors to manage their lifecycle
3. Allow other models to access and use these tensors
4. Automatically handle tensor cleanup when they are no longer needed

```
┌─────────────┐  Creates   ┌─────────────────┐
│  Model A    │───────────▶│  SharedTensor   │
└─────────────┘            └────────┬────────┘
                                    │
                                    │ Uses
                                    ▼
┌─────────────┐            ┌─────────────────┐
│  Model B    │◀───────────│ Reference Count │
└─────────────┘            └─────────────────┘
```

## Supported Tensor Types

The system automatically identifies compatible model combinations for sharing:

| Tensor Type | Compatible Models | Description |
|-------------|------------------|-------------|
| `text_embedding` | BERT, T5, LLAMA, BART | Text embeddings for NLP models |
| `vision_embedding` | ViT, CLIP, DETR | Vision embeddings for image models |
| `audio_embedding` | Whisper, Wav2Vec2, CLAP | Audio embeddings for speech/audio models |
| `vision_text_joint` | CLIP, LLaVA, BLIP | Joint embeddings for multimodal models |
| `audio_text_joint` | CLAP, Whisper-Text | Joint embeddings for audio-text models |

## API Usage

### Creating Shared Tensors

```typescript
// Create a ViT model
const vitModel = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224'
});
await vitModel.initialize();

// Process an image
const imageInput = {
  imageData: preprocessedImage,
  width: 224,
  height: 224
};
const result = await vitModel.process(imageInput);

// Create a shared tensor from the vision embeddings
// (This happens automatically during vitModel.process())
const sharedTensor = vitModel.getSharedTensor('vision_embedding');
```

### Using Shared Tensors

```typescript
// Create a multimodal model that uses vision embeddings
const clipModel = createClipModel(hardware, {
  modelId: 'openai/clip-vit-base-patch16',
  useOptimizedOps: true
});
await clipModel.initialize();

// Process text input 
const textInput = "a photo of a cat";
const textTokens = await clipModel.tokenizeText(textInput);

// Run inference using the shared vision embedding
// This avoids reprocessing the image
const result = await clipModel.processWithSharedEmbedding(
  textTokens,
  sharedTensor
);
```

### Managing Shared Tensor Lifecycle

Shared tensors use reference counting to manage their lifecycle:

```typescript
// The shared tensor is created and maintained by the original model
const sharedTensor = vitModel.getSharedTensor('vision_embedding');

// When using the shared tensor in another model, the reference count increases
clipModel.useSharedTensor(sharedTensor);  // Reference count: 2

// When a model is done with a shared tensor, it should release its reference
clipModel.releaseSharedTensor(sharedTensor);  // Reference count: 1

// When the original model is disposed, it releases its reference
await vitModel.dispose();  // Reference count: 0, tensor is freed
```

In most cases, the SDK handles reference counting automatically, but you can explicitly manage references if needed.

## Performance Benefits

### Memory Usage Comparison

| Scenario | Without Sharing | With Sharing | Memory Saved |
|----------|----------------|--------------|-------------|
| ViT + CLIP | ~150MB | ~110MB | ~40MB (26%) |
| BERT + T5 | ~200MB | ~140MB | ~60MB (30%) |
| Whisper + CLAP | ~120MB | ~90MB | ~30MB (25%) |

### Inference Time Comparison

| Scenario | Without Sharing | With Sharing | Time Saved |
|----------|----------------|--------------|------------|
| ViT + CLIP | ~150ms | ~100ms | ~50ms (33%) |
| BERT + T5 | ~180ms | ~120ms | ~60ms (33%) |
| Whisper + CLAP | ~220ms | ~170ms | ~50ms (23%) |

## Real-World Applications

### Multimodal Vision-Language Models

```typescript
// Process image with ViT
const vitResult = await vitModel.process(imageInput);
const visionEmbedding = vitModel.getSharedTensor('vision_embedding');

// Process text with BERT
const bertResult = await bertModel.process(textInput);
const textEmbedding = bertModel.getSharedTensor('text_embedding');

// Use both embeddings in multimodal model
const multimodalResult = await multimodalModel.processWithSharedEmbeddings({
  visionEmbedding,
  textEmbedding
});
```

### Cascading Models

```typescript
// Run whisper for speech recognition
const transcription = await whisperModel.process(audioInput);
const audioEmbedding = whisperModel.getSharedTensor('audio_embedding');

// Perform sentiment analysis on the transcription using the audio embedding
const sentimentResult = await sentimentModel.processWithSharedEmbedding(
  transcription.text,
  audioEmbedding
);
```

### Streaming Applications

```typescript
// Initialize models
const streamingProcessor = new StreamingProcessor({
  audioModel: whisperModel,
  textModel: bertModel,
  useSharedTensors: true
});

// Process streaming audio
streamingProcessor.onAudioChunk((audioChunk) => {
  // Process audio with shared tensors automatically managed
  const result = streamingProcessor.processAudioWithTranscription(audioChunk);
  
  // Use the result in real-time
  updateUI(result);
});
```

## Browser Compatibility

Cross-model tensor sharing works across all supported browsers and backends:

| Browser | WebGPU Sharing | WebNN Sharing | CPU Sharing |
|---------|---------------|--------------|------------|
| Chrome 113+ | ✅ | ❌ | ✅ |
| Edge 113+ | ✅ | ✅ | ✅ |
| Firefox 118+ | ✅ | ❌ | ✅ |
| Safari 17+ | ✅ | ❌ | ✅ |

## Best Practices

1. **Plan Model Pipeline**: Design your application to reuse intermediate representations where possible
2. **Consider Memory Constraints**: Prioritize sharing for largest tensors to maximize memory savings
3. **Release References**: Explicitly release shared tensors when no longer needed in memory-constrained environments
4. **Monitor Performance**: Measure the impact of sharing on your specific use case
5. **Use Compatible Models**: Pair models that can effectively share the same tensor format

## Debugging

The SDK provides utilities for debugging tensor sharing issues:

```typescript
// Enable verbose logging for shared tensors
SharedTensor.setVerboseLogging(true);

// Get diagnostic information about all active shared tensors
const diagnostics = SharedTensor.getDiagnostics();
console.log(diagnostics);

// Check reference counts
console.log(`Reference count: ${sharedTensor.getReferenceCount()}`);
```

## Advanced Configuration

For fine-grained control over tensor sharing:

```typescript
// Configure which types of tensors can be shared
const config = {
  sharedTensorConfig: {
    enableSharedTensors: true,
    allowedTypes: ['vision_embedding', 'text_embedding'],
    maxSharedTensorSize: 100 * 1024 * 1024 // 100MB max per tensor
  }
};

// Create models with this configuration
const vitModel = createVitModel(hardware, {
  ...config,
  modelId: 'google/vit-base-patch16-224'
});
```

## Example: ViT and BERT Multimodal Integration

```typescript
import { 
  createVitModel, 
  createBertModel,
  createWebGPUBackend 
} from 'ipfs-accelerate-js';

async function runMultimodalAnalysis(imageData, text) {
  // Create hardware backend
  const hardware = await createWebGPUBackend();
  await hardware.initialize();
  
  // Create models
  const vitModel = createVitModel(hardware, {
    modelId: 'google/vit-base-patch16-224',
    useOptimizedOps: true
  });
  
  const bertModel = createBertModel(hardware, {
    modelId: 'bert-base-uncased',
    useOptimizedOps: true
  });
  
  // Initialize models
  await Promise.all([
    vitModel.initialize(),
    bertModel.initialize()
  ]);
  
  try {
    // Process image with ViT
    const vitInput = {
      imageData,
      width: 224,
      height: 224
    };
    const vitResult = await vitModel.process(vitInput);
    
    // Get shared vision embedding
    const visionEmbedding = vitModel.getSharedTensor('vision_embedding');
    
    // Process text with BERT
    const tokens = await bertModel.tokenize(text);
    const bertResult = await bertModel.process(tokens);
    
    // In a real multimodal system, we would now combine the embeddings
    // For this example, we'll just return both results
    return {
      visionResult: vitResult,
      textResult: bertResult,
      // Memory saved by using shared tensor (estimated)
      memorySaved: estimateMemorySaved(visionEmbedding)
    };
  } finally {
    // Clean up resources
    await vitModel.dispose();
    await bertModel.dispose();
  }
}

function estimateMemorySaved(sharedTensor) {
  if (!sharedTensor) return 0;
  
  // Estimate memory saved in MB (assuming float32 values, 4 bytes each)
  const tensorElements = sharedTensor.tensor.dimensions.reduce((a, b) => a * b, 1);
  return (tensorElements * 4) / (1024 * 1024);
}
```

## Further Resources

- [SharedTensor API Reference](./api/shared_tensor.md)
- [Multimodal Examples](../examples/multimodal/)
- [Performance Optimization Guide](./optimization_guide.md)
- [Memory Management Best Practices](./memory_management.md)