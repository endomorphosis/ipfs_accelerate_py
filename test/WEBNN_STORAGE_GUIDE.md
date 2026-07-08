# WebNN Storage Manager User Guide

**Version:** 1.0.0  
**Updated:** March 16, 2025  
**Status:** Complete

## Overview

The WebNN Storage Manager provides persistent storage for model weights and tensors using IndexedDB. It enables efficient caching of AI models for faster loading times and offline operation. This guide explains how to use the Storage Manager and WebNN Storage Integration in your web applications.

## Features

- **Model Caching**: Store model weights for faster loading in future sessions
- **Automatic Cleanup**: Automatically removes unused models to free up space
- **Quota Management**: Monitors and manages storage quotas
- **Versioning**: Supports versioning of models and database schema
- **Fast Access**: Optimized for fast model loading and tensor retrieval
- **Compression**: Framework for compressing large tensors (placeholder implementation)
- **Type Conversion**: Automatic conversion between different tensor data types
- **Metadata Storage**: Comprehensive model metadata storage and retrieval

## Basic Usage

### Initializing the Storage Manager

```typescript
import { StorageManager } from './ipfs_accelerate_js_storage_manager';

// Create a new storage manager instance
const storageManager = new StorageManager({
  dbName: 'my-model-cache',        // Name of the IndexedDB database
  enableCompression: false,        // Enable compression for stored tensors
  maxStorageSize: 1024 * 1024 * 512, // 512MB maximum storage size
  enableAutoCleanup: true,         // Enable automatic cleanup of unused items
  cleanupThreshold: 1000 * 60 * 60 * 24 * 7, // 7 days threshold
  enableLogging: true              // Enable logging for debugging
});

// Initialize the storage manager
const initialized = await storageManager.initialize();
if (!initialized) {
  console.error('Failed to initialize storage manager');
}
```

### Storing Model Weights

```typescript
// Store model metadata
const modelMetadata = {
  id: 'bert-base-uncased',         // Unique model identifier
  name: 'BERT Base Uncased',       // Human-readable name
  version: '1.0.0',                // Version string
  framework: 'tensorflow',         // Optional framework information
  totalSize: 0,                    // Will be calculated automatically
  tensorIds: []                    // Will be populated below
};

// Create a map of tensor names to tensor data
const tensorData = new Map();
tensorData.set('embedding.weight', {
  data: new Float32Array([...]),   // Tensor data
  shape: [30522, 768],             // Tensor shape
  dataType: 'float32' as const     // Data type
});

tensorData.set('encoder.layer.0.attention.self.query.weight', {
  data: new Float32Array([...]),
  shape: [768, 768],
  dataType: 'float32' as const
});

// ... add more tensors as needed

// Calculate total size and tensor IDs
let totalSize = 0;
const tensorIds = [];

for (const [name, tensor] of tensorData.entries()) {
  const tensorId = `${modelMetadata.id}_${name}`;
  tensorIds.push(tensorId);
  totalSize += tensor.data.byteLength;
  
  // Store each tensor
  await storageManager.storeTensor(
    tensorId,
    tensor.data,
    tensor.shape,
    tensor.dataType,
    { name }
  );
}

// Update metadata with size and tensor IDs
modelMetadata.totalSize = totalSize;
modelMetadata.tensorIds = tensorIds;

// Store the model metadata
await storageManager.storeModelMetadata(modelMetadata);
```

### Loading Models

```typescript
// Check if a model exists
const modelId = 'bert-base-uncased';
const modelMetadata = await storageManager.getModelMetadata(modelId);

if (modelMetadata) {
  console.log(`Found model: ${modelMetadata.name} (v${modelMetadata.version})`);
  console.log(`Total size: ${formatBytes(modelMetadata.totalSize)}`);
  
  // Load each tensor
  const tensors = new Map();
  for (const tensorId of modelMetadata.tensorIds) {
    const tensorData = await storageManager.getTensorData(tensorId);
    const tensor = await storageManager.getTensor(tensorId);
    
    if (tensorData && tensor) {
      // Extract tensor name from metadata
      const name = tensor.metadata?.name || tensorId.replace(`${modelId}_`, '');
      tensors.set(name, {
        data: tensorData,
        shape: tensor.shape
      });
    }
  }
  
  console.log(`Loaded ${tensors.size} tensors`);
  // Use the tensors for inference...
} else {
  console.log(`Model '${modelId}' not found in storage`);
}

// Helper function to format bytes
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
```

### Managing Models

```typescript
// List all stored models
const models = await storageManager.listModels();
console.log(`Found ${models.length} models in storage:`);

for (const model of models) {
  console.log(`- ${model.name} (${model.id}, v${model.version})`);
  console.log(`  Size: ${formatBytes(model.totalSize)}`);
  console.log(`  Last accessed: ${new Date(model.lastAccessed).toLocaleString()}`);
}

// Delete a model
const deleted = await storageManager.deleteModel('bert-base-uncased');
if (deleted) {
  console.log('Model deleted successfully');
} else {
  console.log('Failed to delete model or model not found');
}

// Clear all models
const cleared = await storageManager.clear();
if (cleared) {
  console.log('All models cleared from storage');
} else {
  console.log('Failed to clear storage');
}

// Get storage statistics
const stats = await storageManager.getStorageInfo();
console.log(`Storage statistics:`);
console.log(`- Model count: ${stats.modelCount}`);
console.log(`- Tensor count: ${stats.tensorCount}`);
console.log(`- Total size: ${formatBytes(stats.totalSize)}`);
if (stats.remainingQuota) {
  console.log(`- Remaining quota: ${formatBytes(stats.remainingQuota)}`);
}
```

## Using the WebNN Storage Integration

The WebNN Storage Integration provides a higher-level interface for using the Storage Manager with the WebNN backend:

```typescript
import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import { WebNNStorageIntegration } from './ipfs_accelerate_js_webnn_storage_integration';

// Create and initialize WebNN backend
const backend = new WebNNBackend({
  enableLogging: true,
  deviceType: 'gpu'
});
await backend.initialize();

// Create storage integration
const storage = new WebNNStorageIntegration(backend, {
  enableLogging: true,
  enableModelCaching: true,
  dbName: 'webnn-model-cache'
});
await storage.initialize();
```

### Storing Models with the Integration

```typescript
// Create model weights
const weights = new Map();

// Add weights tensors
weights.set('weights', {
  data: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
  shape: [4, 2],
  dataType: 'float32' as const
});

weights.set('bias', {
  data: new Float32Array([0.1, 0.2]),
  shape: [1, 2],
  dataType: 'float32' as const
});

// Store model
const modelId = 'simple-mlp';
const modelName = 'Simple MLP Model';
const stored = await storage.storeModel(
  modelId,
  modelName,
  weights,
  {
    version: '1.0.0',
    framework: 'custom',
    description: 'A simple MLP model for demonstration'
  }
);

if (stored) {
  console.log('Model stored successfully');
} else {
  console.log('Failed to store model');
}
```

### Loading Models and Running Inference

```typescript
// Check if model is cached
const isCached = await storage.isModelCached(modelId);
console.log(`Model "${modelName}" is${isCached ? '' : ' not'} cached`);

if (isCached) {
  // Load model tensors
  const tensors = await storage.loadModel(modelId);
  
  if (tensors) {
    console.log(`Loaded ${tensors.size} tensors from cache`);
    
    // Create input tensor
    const input = await backend.createTensor(
      new Float32Array([1, 2, 3, 4]),
      [1, 4],
      'float32'
    );
    
    // Use the loaded weights for inference
    const weights = tensors.get('weights');
    const bias = tensors.get('bias');
    
    // Matrix multiplication: input × weights
    const result1 = await backend.execute('matmul', {
      a: input,
      b: weights
    });
    
    // Add bias
    const result2 = await backend.execute('add', {
      a: result1,
      b: bias
    });
    
    // Read result
    const output = await backend.readTensor(result2.tensor, result2.shape);
    console.log('Inference result:', Array.from(output as Float32Array));
  }
}
```

### Managing Models with the Integration

```typescript
// Get storage statistics
const stats = await storage.getStorageStats();
console.log('Storage statistics:', stats);

// List all models
const models = await storage.listModels();
console.log('Available models:');
for (const model of models) {
  console.log(`- ${model.name} (${model.id}, v${model.version}, ${formatBytes(model.size)})`);
}

// Delete a model
const deleted = await storage.deleteModel(modelId);
if (deleted) {
  console.log(`Model ${modelId} deleted`);
} else {
  console.log(`Failed to delete model ${modelId}`);
}

// Clear cache
const cleared = await storage.clearCache();
if (cleared) {
  console.log('Cache cleared successfully');
} else {
  console.log('Failed to clear cache');
}
```

## Advanced Usage

### Compression Configuration

The Storage Manager includes placeholders for compression (to be fully implemented in the future):

```typescript
const storageManager = new StorageManager({
  enableCompression: true,
  // Additional compression options could be added here in the future
});
```

### Custom Cleanup Strategies

Control how unused tensors are cleaned up:

```typescript
const storageManager = new StorageManager({
  enableAutoCleanup: true,
  cleanupThreshold: 1000 * 60 * 60 * 24 * 30, // 30 days (keep longer)
});

// Manually trigger cleanup
const freedSpace = await storageManager.cleanup();
console.log(`Freed ${formatBytes(freedSpace)} of space`);
```

### Browser Storage Quotas

The Storage Manager can detect and respect browser storage quotas:

```typescript
// Get storage information including remaining quota
const storageInfo = await storageManager.getStorageInfo();

if (storageInfo.remainingQuota) {
  // We have quota information
  const usedPercentage = (storageInfo.totalSize / 
    (storageInfo.totalSize + storageInfo.remainingQuota)) * 100;
  
  console.log(`Using ${usedPercentage.toFixed(2)}% of available storage`);
  
  if (usedPercentage > 80) {
    console.warn('Storage usage is high, consider clearing unused models');
  }
}
```

## Performance Considerations

### Optimizing Storage Access

- **Batch Operations**: When storing multiple tensors, store them in a single batch operation if possible
- **Access Patterns**: Group related tensors together with similar naming conventions
- **Data Types**: Use the most memory-efficient data type for your tensors (e.g., 'float16' instead of 'float32' when precision is less critical)
- **Cleanup Strategy**: Adjust cleanup threshold based on your application's needs

### Memory Management

- **Tensor Data Copies**: Be aware that `getTensorData()` creates a new copy of the tensor data
- **Large Models**: For very large models, consider loading tensors only when needed rather than all at once
- **Garbage Collection**: The browser will eventually garbage collect unused ArrayBuffers, but you can help by nullifying references when done

## Error Handling

The Storage Manager methods return Promises, so use try/catch blocks for error handling:

```typescript
try {
  const model = await storageManager.getModelMetadata('non-existent-model');
  if (!model) {
    console.log('Model not found');
  }
} catch (error) {
  console.error('Error accessing storage:', error);
  
  if (error.name === 'QuotaExceededError') {
    console.error('Storage quota exceeded. Try clearing some space.');
  }
}
```

## Browser Compatibility

The Storage Manager uses IndexedDB, which is supported in all modern browsers:

| Browser | IndexedDB Support | Notes |
|---------|-------------------|-------|
| Chrome  | Full              | Best performance |
| Edge    | Full              | Based on Chromium, good performance |
| Firefox | Full              | Good performance |
| Safari  | Full              | Improved in recent versions |

Safari historically had issues with IndexedDB, but recent versions have improved compatibility.

## Conclusion

The WebNN Storage Manager provides a powerful system for caching and managing model weights in the browser. By using this system, you can significantly improve the loading time for models and enable offline operation for your AI-powered web applications.

For additional information and advanced usage scenarios, see the API documentation and example applications.