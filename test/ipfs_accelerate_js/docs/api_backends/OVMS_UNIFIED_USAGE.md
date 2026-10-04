# OpenVINO Model Server (OVMS) Unified Backend

The OVMS Unified backend provides a comprehensive interface for working with OpenVINO Model Server, enabling inference for computer vision, natural language processing, and other AI model types through both hosted API and containerized deployment options.

## Introduction

The OVMS Unified backend extends the standard OVMS backend with advanced features for handling both API and container deployments:

- **Dual-mode operation**: Seamlessly switch between API and container modes
- **Comprehensive model management**: Deploy, configure, and monitor a wide range of AI models
- **Advanced hardware acceleration**: Optimized for Intel CPUs, GPUs, and specialized AI hardware
- **Model versioning**: Support for multiple model versions and A/B testing
- **Batch inference**: High-throughput processing for optimal performance
- **Quantization support**: INT8/INT4 precision options for improved efficiency
- **Execution mode selection**: Optimize for latency or throughput based on requirements
- **Circuit breaker pattern**: Prevents cascading failures with automatic recovery
- **Request prioritization**: Assign importance levels to different inference requests
- **Robust error handling**: Typed error classification with exponential backoff
- **Performance benchmarking**: Tools for measuring throughput and optimizing configurations
- **Container lifecycle management**: Deploy and manage containerized model servers

## Prerequisites

For using the OVMS Unified backend, you'll need:

1. For API mode:
   - API endpoint URL
   - API key (optional, depending on deployment security)
   - Model name/ID

2. For container mode:
   - Docker installed on your system
   - Sufficient system resources for model deployment
   - Intel hardware (CPU/GPU/specialized accelerators) for optimal performance

## Installation

The OVMS Unified backend is included in the IPFS Accelerate JavaScript SDK:

```bash
npm install ipfs-accelerate
```

## Basic Usage

### Initializing the Backend

```typescript
import { OVMSUnified } from 'ipfs-accelerate/api_backends/ovms_unified';
import { ApiResources, ApiMetadata } from 'ipfs-accelerate/api_backends/types';

// Create resources object (optional)
const resources: ApiResources = {};

// Set up metadata with API connection information
const metadata: ApiMetadata = {
  ovms_api_url: 'http://localhost:9000',
  ovms_api_key: 'YOUR_API_KEY', // Optional, may not be required for local deployments
  ovms_model: 'resnet50',
  ovms_version: 'latest',
  ovms_precision: 'FP32',
  timeout: 30000 // 30 second timeout
};

// Create the OVMS Unified backend instance
const ovmsUnified = new OVMSUnified(resources, metadata);

console.log(`Initialized in ${ovmsUnified.getMode()} mode`);
```

### Basic Inference

```typescript
// Basic inference with a simple input
async function runInference() {
  // Input data - format depends on your model's requirements
  const inputData = Array.from({ length: 3 * 224 * 224 }, () => Math.random());
  
  // Run inference with default model
  try {
    const result = await ovmsUnified.infer(undefined, inputData);
    console.log('Inference result:', result);
    
    return result;
  } catch (error) {
    console.error('Inference error:', error);
    throw error;
  }
}
```

### Structured Input Inference

```typescript
// Inference with structured input
async function runStructuredInference() {
  // Create structured input with multiple fields
  const structuredInput = {
    data: Array.from({ length: 3 * 224 * 224 }, () => Math.random()),
    parameters: {
      scale: 0.5,
      normalize: true
    }
  };
  
  try {
    const result = await ovmsUnified.infer('resnet50', structuredInput);
    console.log('Structured inference result:', result);
    
    return result;
  } catch (error) {
    console.error('Structured inference error:', error);
    throw error;
  }
}
```

### Batch Inference

```typescript
// Batch inference for higher throughput
async function runBatchInference() {
  // Create a batch of inputs
  const batchInputs = [
    Array.from({ length: 3 * 224 * 224 }, () => Math.random()),
    Array.from({ length: 3 * 224 * 224 }, () => Math.random()),
    Array.from({ length: 3 * 224 * 224 }, () => Math.random())
  ];
  
  try {
    // Run batch inference
    const results = await ovmsUnified.batchInfer('resnet50', batchInputs);
    console.log(`Processed ${results.length} inputs in batch`);
    
    return results;
  } catch (error) {
    console.error('Batch inference error:', error);
    throw error;
  }
}
```

## Model Management

### Getting Model Information

```typescript
// Get detailed model information
async function getModelMetadata() {
  try {
    const modelInfo = await ovmsUnified.getModelInfo('resnet50');
    
    console.log(`Model: ${modelInfo.name}`);
    console.log(`Platform: ${modelInfo.platform}`);
    console.log(`Available versions: ${modelInfo.versions.join(', ')}`);
    
    // Display input shapes
    modelInfo.inputs.forEach(input => {
      console.log(`Input: ${input.name}, shape: [${input.shape.join(', ')}], type: ${input.datatype}`);
    });
    
    // Display output shapes
    modelInfo.outputs.forEach(output => {
      console.log(`Output: ${output.name}, shape: [${output.shape.join(', ')}], type: ${output.datatype}`);
    });
    
    return modelInfo;
  } catch (error) {
    console.error('Error getting model info:', error);
    throw error;
  }
}
```

### Working with Model Versions

```typescript
// List and use different model versions
async function workWithModelVersions() {
  try {
    // Get available versions
    const versions = await ovmsUnified.getModelVersions('resnet50');
    console.log(`Available versions: ${versions.join(', ')}`);
    
    // Test with a specific version
    if (versions.length > 0) {
      const version = versions[0];
      console.log(`Running inference with version ${version}`);
      
      const inputData = Array.from({ length: 3 * 224 * 224 }, () => Math.random());
      const result = await ovmsUnified.inferWithVersion('resnet50', version, inputData);
      
      console.log(`Version ${version} result:`, result);
      return result;
    }
  } catch (error) {
    console.error('Error working with model versions:', error);
    throw error;
  }
}
```

## Performance Optimization

### Configuring Model for Performance

```typescript
// Configure model for optimal performance
async function optimizeModelPerformance() {
  try {
    // Get server statistics before optimization
    const initialStats = await ovmsUnified.getServerStatistics();
    console.log('Initial server statistics:');
    console.log(`- Average inference time: ${initialStats.avg_inference_time} ms`);
    console.log(`- Requests per second: ${initialStats.requests_per_second}`);
    
    // Configure model for throughput optimization
    await ovmsUnified.setModelConfig('resnet50', {
      batch_size: 32,            // Maximum batch size
      preferred_batch: 16,       // Preferred batch size
      instance_count: 2,         // Number of model instances
      execution_mode: 'throughput' // Optimize for throughput
    });
    console.log('Model configured for throughput optimization');
    
    // Run benchmark with throughput configuration
    const inputData = Array.from({ length: 3 * 224 * 224 }, () => Math.random());
    const batchInputs = Array(16).fill(inputData);
    
    console.log('Running batch inference...');
    const startTime = performance.now();
    const results = await ovmsUnified.batchInfer('resnet50', batchInputs);
    const endTime = performance.now();
    
    console.log(`Processed ${results.length} inputs in ${(endTime - startTime).toFixed(2)} ms`);
    console.log(`Throughput: ${(results.length / (endTime - startTime) * 1000).toFixed(2)} inputs/second`);
    
    // Switch to latency optimization
    await ovmsUnified.setExecutionMode('resnet50', 'latency');
    console.log('Model configured for latency optimization');
    
    // Run single inference with latency configuration
    const latencyStart = performance.now();
    await ovmsUnified.infer('resnet50', inputData);
    const latencyEnd = performance.now();
    
    console.log(`Single inference time: ${(latencyEnd - latencyStart).toFixed(2)} ms`);
    
    // Get server statistics after optimization
    const finalStats = await ovmsUnified.getServerStatistics();
    console.log('Final server statistics:');
    console.log(`- Average inference time: ${finalStats.avg_inference_time} ms`);
    console.log(`- Requests per second: ${finalStats.requests_per_second}`);
    
    return {
      throughputResults: results.length,
      throughputTime: endTime - startTime,
      latencyTime: latencyEnd - latencyStart,
      initialStats,
      finalStats
    };
  } catch (error) {
    console.error('Error optimizing model performance:', error);
    throw error;
  }
}
```

### Quantization Configuration

```typescript
// Configure model quantization
async function configureQuantization() {
  try {
    // Set up INT8 quantization
    await ovmsUnified.setQuantization('resnet50', {
      enabled: true,
      method: 'MinMax',
      bits: 8
    });
    console.log('INT8 quantization enabled');
    
    // Run inference with quantized model
    const inputData = Array.from({ length: 3 * 224 * 224 }, () => Math.random());
    const result = await ovmsUnified.infer('resnet50', inputData);
    console.log('Quantized model inference result:', result);
    
    // Disable quantization
    await ovmsUnified.setQuantization('resnet50', {
      enabled: false
    });
    console.log('Quantization disabled');
    
    return result;
  } catch (error) {
    console.error('Error configuring quantization:', error);
    throw error;
  }
}
```

## Switching Between API and Container Modes

### API Mode

```typescript
// Set up API mode
async function setupApiMode() {
  // Set up metadata with API connection information
  const metadata: ApiMetadata = {
    ovms_api_url: 'http://your-api-server.com:9000',
    ovms_api_key: 'YOUR_API_KEY',
    ovms_model: 'resnet50'
  };
  
  // Create backend in API mode
  const ovmsUnified = new OVMSUnified({}, metadata);
  ovmsUnified.setMode(false); // Explicitly set to API mode
  
  console.log(`Mode: ${ovmsUnified.getMode()}`);
  
  // Test endpoint availability
  const isAvailable = await ovmsUnified.testEndpoint();
  console.log(`API endpoint available: ${isAvailable}`);
  
  return ovmsUnified;
}
```

### Container Mode

```typescript
// Set up container mode
async function setupContainerMode() {
  // Create backend in container mode
  const ovmsUnified = new OVMSUnified({
    useContainer: true
  }, {
    ovms_model: 'resnet50'
  });
  
  // Explicitly set to container mode
  ovmsUnified.setMode(true);
  console.log(`Mode: ${ovmsUnified.getMode()}`);
  
  // Configure container deployment
  const deploymentConfig = {
    dockerRegistry: 'openvino/model_server',
    containerTag: 'latest',
    gpuDevice: '0', // Use first GPU, or '' for CPU-only
    modelId: 'resnet50',
    port: 9000,
    env: {
      'ALLOW_ANONYMOUS_REQUESTS': '1'
    },
    volumes: ['./models:/models'],
    maxInputLength: 8192
  };
  
  // Start container
  try {
    const containerInfo = await ovmsUnified.startContainer(deploymentConfig);
    console.log('Container started:');
    console.log(`- Container ID: ${containerInfo.containerId}`);
    console.log(`- Host: ${containerInfo.host}`);
    console.log(`- Port: ${containerInfo.port}`);
    
    return ovmsUnified;
  } catch (error) {
    console.error('Error starting container:', error);
    throw error;
  }
}
```

### Stopping Container

```typescript
// Stop container when done
async function stopContainer(ovmsUnified) {
  try {
    const stopped = await ovmsUnified.stopContainer();
    console.log(`Container stopped: ${stopped}`);
    return stopped;
  } catch (error) {
    console.error('Error stopping container:', error);
    throw error;
  }
}
```

## Error Handling and Resilience

### Basic Error Handling

```typescript
// Basic error handling
async function handleInferenceErrors() {
  try {
    // Try inference with a model that might not exist
    const result = await ovmsUnified.infer('nonexistent_model', [1, 2, 3, 4, 5]);
    console.log('Inference result:', result);
    return result;
  } catch (error) {
    if (error.message.includes('not found')) {
      console.error('Model not found error. Please check the model name or ensure it is loaded.');
    } else if (error.message.includes('timeout')) {
      console.error('Request timed out. The server might be busy or the model is too large.');
    } else if (error.message.includes('connection')) {
      console.error('Connection error. Check that the OVMS server is running.');
    } else {
      console.error('Unexpected error:', error);
    }
    
    // Provide a fallback response or error code
    return {
      error: error.message,
      errorType: error.name,
      status: 'failed'
    };
  }
}
```

### Robust Error Handling with Retries

```typescript
// Advanced error handling with exponential backoff
async function robustInference(model, data, maxRetries = 3) {
  let retries = 0;
  let lastError = null;
  
  while (retries <= maxRetries) {
    try {
      return await ovmsUnified.infer(model, data);
    } catch (error) {
      lastError = error;
      
      // Check if error is retriable
      if (
        error.message.includes('timeout') ||
        error.message.includes('connection') ||
        error.message.includes('busy') ||
        error.message.includes('temporary')
      ) {
        retries++;
        console.log(`Retriable error detected. Retry ${retries}/${maxRetries}`);
        
        // Exponential backoff
        const delay = Math.pow(2, retries) * 100;
        console.log(`Waiting ${delay}ms before retry...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      } else {
        // Non-retriable error
        console.log('Non-retriable error detected. Aborting.');
        throw error;
      }
    }
  }
  
  // Exhausted retries
  throw new Error(`Failed after ${maxRetries} retries: ${lastError?.message}`);
}
```

### Circuit Breaker Pattern

```typescript
// Example usage showing circuit breaker behavior
async function demonstrateCircuitBreaker() {
  // Set up a backend with circuit breaker (usually handled internally)
  const backend = new OVMSUnified({
    circuitBreakerThreshold: 3,   // Trip after 3 consecutive failures
    circuitBreakerResetTime: 5000 // 5 seconds of cooling period
  }, {
    ovms_api_url: 'http://localhost:9000',
    ovms_model: 'resnet50'
  });
  
  // Simulate multiple failures and observe circuit breaker behavior
  console.log('Testing circuit breaker pattern:');
  
  // Mock the infer method to always fail for this demonstration
  const originalInfer = backend.infer;
  backend.infer = async () => {
    throw new Error('Simulated server error');
  };
  
  for (let i = 0; i < 5; i++) {
    try {
      await backend.infer('resnet50', [1, 2, 3, 4, 5]);
    } catch (error) {
      console.log(`Request ${i+1} error: ${error.message}`);
      
      // Check circuit state (implementation detail)
      if (error.message.includes('circuit breaker')) {
        console.log('Circuit breaker opened - requests are blocked');
      }
    }
  }
  
  // Wait for circuit breaker to reset
  console.log('Waiting for circuit breaker reset...');
  await new Promise(resolve => setTimeout(resolve, 5000));
  
  // Try again after reset
  try {
    await backend.infer('resnet50', [1, 2, 3, 4, 5]);
  } catch (error) {
    console.log(`After reset error: ${error.message}`);
  }
  
  // Restore original behavior
  backend.infer = originalInfer;
}
```

## Performance Benchmarking

```typescript
// Benchmark inference performance
async function runPerformanceBenchmark() {
  try {
    // Create sample data for benchmarking
    const inputData = Array.from({ length: 3 * 224 * 224 }, () => Math.random());
    
    // Set up benchmark options
    const benchmarkOptions = {
      iterations: 10,
      batchSizes: [1, 4, 8, 16],
      model: 'resnet50',
      timeout: 60000,
      warmupIterations: 2
    };
    
    console.log('Starting benchmark...');
    console.log(`Model: ${benchmarkOptions.model}`);
    console.log(`Iterations: ${benchmarkOptions.iterations}`);
    
    const results = {};
    
    // Run benchmarks for each batch size
    for (const batchSize of benchmarkOptions.batchSizes) {
      console.log(`\nBenchmarking batch size: ${batchSize}`);
      
      // Create batch input
      const batchInput = Array(batchSize).fill(inputData);
      
      // Warmup
      console.log('Warming up...');
      for (let i = 0; i < benchmarkOptions.warmupIterations; i++) {
        await ovmsUnified.batchInfer(benchmarkOptions.model, batchInput);
      }
      
      // Benchmark
      console.log('Running benchmark...');
      const startTime = performance.now();
      
      for (let i = 0; i < benchmarkOptions.iterations; i++) {
        await ovmsUnified.batchInfer(benchmarkOptions.model, batchInput);
      }
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const averageTime = totalTime / benchmarkOptions.iterations;
      const itemsPerSecond = (batchSize * benchmarkOptions.iterations) / (totalTime / 1000);
      
      results[batchSize] = {
        totalTime,
        averageTime,
        itemsPerSecond
      };
      
      console.log(`- Total time: ${totalTime.toFixed(2)} ms`);
      console.log(`- Average time per batch: ${averageTime.toFixed(2)} ms`);
      console.log(`- Items per second: ${itemsPerSecond.toFixed(2)}`);
    }
    
    // Calculate speedup from batch processing
    if (results[1]) {
      const baselinePerformance = results[1].itemsPerSecond;
      
      for (const batchSize of benchmarkOptions.batchSizes.slice(1)) {
        const speedup = results[batchSize].itemsPerSecond / baselinePerformance;
        results[batchSize].speedup = speedup;
        console.log(`\nBatch size ${batchSize} speedup: ${speedup.toFixed(2)}x`);
      }
    }
    
    return results;
  } catch (error) {
    console.error('Benchmark error:', error);
    throw error;
  }
}
```

## Practical Applications

### Image Classification

```typescript
// Image classification application
async function classifyImage(imageData) {
  try {
    // Preprocess image data (normalize, resize, etc.)
    const preprocessedData = preprocessImage(imageData); // Implement your preprocessing logic
    
    // Run inference
    const result = await ovmsUnified.infer('resnet50', preprocessedData);
    
    // Process results (find top classes)
    const classes = postprocessClassification(result); // Implement your postprocessing logic
    
    return classes;
  } catch (error) {
    console.error('Image classification error:', error);
    throw error;
  }
}

// Example preprocessing function
function preprocessImage(imageData) {
  // Convert image to the right format, normalize, etc.
  // This is a simplified example
  return Array.from({ length: 3 * 224 * 224 }, (_, i) => {
    // Simple normalization to [0,1] range
    return imageData[i] / 255.0;
  });
}

// Example postprocessing function
function postprocessClassification(result) {
  // Extract predictions
  const predictions = result.predictions || [];
  
  if (predictions.length === 0) {
    return [];
  }
  
  // Find top 5 classes
  const scores = predictions[0];
  
  // Create (index, score) pairs
  const indexedScores = scores.map((score, index) => ({ index, score }));
  
  // Sort by score in descending order
  indexedScores.sort((a, b) => b.score - a.score);
  
  // Take top 5
  const top5 = indexedScores.slice(0, 5);
  
  // Map to class names (you would have your own class list)
  const classNames = [
    'apple', 'banana', 'carrot', 'dog', 'elephant',
    'fish', 'giraffe', 'house', 'igloo', 'jacket'
  ];
  
  return top5.map(item => ({
    className: classNames[item.index % classNames.length], // Simplified mapping
    probability: item.score
  }));
}
```

### Object Detection

```typescript
// Object detection application
async function detectObjects(imageData) {
  try {
    // Preprocess image data
    const preprocessedData = preprocessImageForDetection(imageData); // Implement preprocessing
    
    // Create structured input
    const modelInput = {
      instances: [
        {
          data: preprocessedData
        }
      ]
    };
    
    // Run inference
    const result = await ovmsUnified.infer('yolo', modelInput);
    
    // Process detection results
    const detections = postprocessDetections(result, imageData.width, imageData.height);
    
    return detections;
  } catch (error) {
    console.error('Object detection error:', error);
    throw error;
  }
}

// Simple postprocessing function for detections
function postprocessDetections(result, imageWidth, imageHeight) {
  // This is a simplified example - real implementation would depend on the model output format
  
  // Extract predictions
  const predictions = result.predictions || [];
  
  if (predictions.length === 0) {
    return [];
  }
  
  // Map model outputs to detection format
  // Model output format varies by model (e.g., YOLO, SSD, Faster R-CNN)
  // This is a simplified example assuming output is [x, y, width, height, confidence, class1_score, class2_score, ...]
  return predictions.map(detection => {
    const [x, y, width, height, confidence, ...classScores] = detection;
    
    // Find class with highest score
    const classIndex = classScores.indexOf(Math.max(...classScores));
    
    // Map to class name (you would have your own class list)
    const classNames = ['person', 'car', 'dog', 'cat', 'bicycle'];
    const className = classNames[classIndex % classNames.length];
    
    // Scale coordinates to image dimensions
    return {
      bbox: {
        x: x * imageWidth,
        y: y * imageHeight,
        width: width * imageWidth,
        height: height * imageHeight
      },
      class: className,
      confidence: confidence
    };
  });
}
```

### Semantic Segmentation

```typescript
// Semantic segmentation application
async function performSegmentation(imageData) {
  try {
    // Preprocess image data
    const preprocessedData = preprocessImageForSegmentation(imageData); // Implement preprocessing
    
    // Run inference
    const result = await ovmsUnified.infer('segmentation_model', preprocessedData);
    
    // Process segmentation results
    const segmentationMap = postprocessSegmentation(result, imageData.width, imageData.height);
    
    return segmentationMap;
  } catch (error) {
    console.error('Segmentation error:', error);
    throw error;
  }
}

// Simple postprocessing function for segmentation
function postprocessSegmentation(result, width, height) {
  // Extract predictions
  const predictions = result.predictions || [];
  
  if (predictions.length === 0) {
    return null;
  }
  
  // Map model outputs to segmentation format
  // This is a simplified example - real implementation would depend on model output format
  const rawSegmentation = predictions[0];
  
  // Convert to class indices (assuming channel-first output with one-hot encoding)
  // In a real application, you'd handle this based on your model's specific output format
  const segmentationMap = [];
  
  // Class names (you would have your own class list)
  const classNames = ['background', 'person', 'car', 'road', 'sky', 'vegetation'];
  
  // Create visualization info to return
  return {
    width,
    height,
    segmentationMap: rawSegmentation, // Simplified - would process this in a real app
    classNames
  };
}
```

### API Key Multiplexing

```typescript
// API key multiplexing for load balancing
async function setupApiKeyMultiplexing() {
  // Set up with multiple API keys
  const apiKeys = [
    { key: 'API_KEY_1', priority: 'HIGH' },
    { key: 'API_KEY_2', priority: 'MEDIUM' },
    { key: 'API_KEY_3', priority: 'LOW' }
  ];
  
  // Initialize backend with API key multiplexing
  const backend = new OVMSUnified({
    apiKeyMultiplexing: true,
    apiKeys: apiKeys,
    apiKeySelectionStrategy: 'round-robin' // 'round-robin', 'priority', 'random'
  }, {
    ovms_api_url: 'http://localhost:9000',
    ovms_model: 'resnet50'
  });
  
  // Run multiple inferences to demonstrate key rotation
  console.log('Running multiple inferences with API key rotation:');
  
  const inputData = [1, 2, 3, 4, 5];
  
  for (let i = 0; i < 5; i++) {
    try {
      await backend.infer('resnet50', inputData);
      console.log(`Inference ${i+1} succeeded`);
    } catch (error) {
      console.error(`Inference ${i+1} failed:`, error);
    }
  }
}
```

## Complete Examples

### Basic Usage Example

```typescript
import { OVMSUnified } from 'ipfs-accelerate/api_backends/ovms_unified';

async function basicExample() {
  try {
    // Initialize the backend
    const backend = new OVMSUnified({}, {
      ovms_api_url: 'http://localhost:9000',
      ovms_model: 'resnet50'
    });
    
    // Test if the endpoint is available
    const isAvailable = await backend.testEndpoint();
    console.log(`Endpoint available: ${isAvailable}`);
    
    if (!isAvailable) {
      console.log('Endpoint not available. Please check your OVMS server.');
      return;
    }
    
    // Get model information
    const modelInfo = await backend.getModelInfo();
    console.log('Model information:');
    console.log(`- Name: ${modelInfo.name}`);
    console.log(`- Platform: ${modelInfo.platform}`);
    console.log(`- Versions: ${modelInfo.versions.join(', ')}`);
    
    // Create sample input data
    const inputData = Array.from({ length: 10 }, (_, i) => i);
    
    // Run inference
    console.log('Running inference...');
    const result = await backend.infer('resnet50', inputData);
    
    console.log('Inference result:');
    console.log(JSON.stringify(result, null, 2));
    
    // Get server statistics
    const stats = await backend.getServerStatistics();
    console.log('Server statistics:');
    console.log(`- Uptime: ${stats.server_uptime} seconds`);
    console.log(`- Version: ${stats.server_version}`);
    console.log(`- Active models: ${stats.active_models}`);
    console.log(`- Requests per second: ${stats.requests_per_second}`);
    
    return {
      modelInfo,
      result,
      stats
    };
  } catch (error) {
    console.error('Error in basic example:', error);
    throw error;
  }
}

// Run the example
basicExample().catch(console.error);
```

### Advanced Container Mode Example

```typescript
import { OVMSUnified } from 'ipfs-accelerate/api_backends/ovms_unified';

async function containerModeExample() {
  let backend = null;
  
  try {
    // Initialize in container mode
    backend = new OVMSUnified({
      useContainer: true
    }, {
      ovms_model: 'resnet50'
    });
    
    // Container deployment configuration
    const deployConfig = {
      dockerRegistry: 'openvino/model_server',
      containerTag: 'latest',
      gpuDevice: '', // CPU-only mode for example
      modelId: 'resnet50',
      port: 9000,
      env: {
        'ALLOW_ANONYMOUS_REQUESTS': '1'
      },
      volumes: ['./models:/models'],
      maxInputLength: 8192
    };
    
    // Start container
    console.log('Starting container...');
    const containerInfo = await backend.startContainer(deployConfig);
    
    console.log('Container started:');
    console.log(`- Container ID: ${containerInfo.containerId}`);
    console.log(`- Host: ${containerInfo.host}`);
    console.log(`- Port: ${containerInfo.port}`);
    
    // Wait for model to load
    console.log('Waiting for model to load...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Test if model is available
    const isAvailable = await backend.testEndpoint();
    console.log(`Model available: ${isAvailable}`);
    
    if (isAvailable) {
      // Run inference
      const inputData = Array.from({ length: 10 }, (_, i) => i);
      
      console.log('Running inference...');
      const result = await backend.infer('resnet50', inputData);
      
      console.log('Inference result:');
      console.log(JSON.stringify(result, null, 2));
    }
    
    // Configure model for better performance
    console.log('Configuring model for better performance...');
    await backend.setModelConfig('resnet50', {
      batch_size: 8,
      instance_count: 2,
      execution_mode: 'throughput'
    });
    
    // Create sample batch
    const batchData = Array.from({ length: 5 }, () => 
      Array.from({ length: 10 }, (_, i) => i)
    );
    
    // Run batch inference
    console.log('Running batch inference...');
    const batchResults = await backend.batchInfer('resnet50', batchData);
    
    console.log(`Batch results: ${batchResults.length} results`);
    
    return {
      containerInfo,
      batchResults
    };
  } catch (error) {
    console.error('Error in container example:', error);
    throw error;
  } finally {
    // Always stop the container when done
    if (backend) {
      console.log('Stopping container...');
      try {
        await backend.stopContainer();
        console.log('Container stopped successfully');
      } catch (stopError) {
        console.error('Error stopping container:', stopError);
      }
    }
  }
}

// Run the example
containerModeExample().catch(console.error);
```

### Practical Computer Vision Pipeline

```typescript
import { OVMSUnified } from 'ipfs-accelerate/api_backends/ovms_unified';

async function computerVisionPipeline() {
  try {
    // Initialize the backend
    const backend = new OVMSUnified({}, {
      ovms_api_url: 'http://localhost:9000',
      ovms_model: 'resnet50'
    });
    
    // Create a simple image processing pipeline
    const pipeline = {
      // Step 1: Image classification
      async classify(imageData) {
        // Preprocess image
        const preprocessed = this.preprocess(imageData);
        
        // Run inference
        const result = await backend.infer('resnet50', preprocessed);
        
        // Get top classes
        return this.getTopClasses(result);
      },
      
      // Step 2: Object detection
      async detect(imageData) {
        // Preprocess image
        const preprocessed = this.preprocess(imageData);
        
        // Run detection
        const result = await backend.infer('detection_model', preprocessed);
        
        // Extract detections
        return this.extractDetections(result);
      },
      
      // Step 3: Image segmentation
      async segment(imageData) {
        // Preprocess image
        const preprocessed = this.preprocess(imageData);
        
        // Run segmentation
        const result = await backend.infer('segmentation_model', preprocessed);
        
        // Create segmentation map
        return this.createSegmentationMap(result);
      },
      
      // Helper methods
      preprocess(imageData) {
        // Simplified preprocessing
        return Array.from({ length: 3 * 224 * 224 }, (_, i) => i / (3 * 224 * 224));
      },
      
      getTopClasses(result) {
        // Mock class list
        const classes = ['apple', 'banana', 'carrot', 'dog', 'elephant'];
        
        // Extract top classes (simplified)
        const scores = result.predictions?.[0] || [0.1, 0.2, 0.5, 0.1, 0.1];
        const topIdx = scores.indexOf(Math.max(...scores));
        
        return {
          className: classes[topIdx],
          confidence: scores[topIdx]
        };
      },
      
      extractDetections(result) {
        // Simplified detection extraction
        return [
          { class: 'person', confidence: 0.92, bbox: [10, 20, 100, 200] },
          { class: 'car', confidence: 0.87, bbox: [150, 30, 250, 180] }
        ];
      },
      
      createSegmentationMap(result) {
        // Simplified segmentation map
        return {
          width: 224,
          height: 224,
          classes: ['background', 'person', 'car', 'road', 'sky'],
          data: new Uint8Array(224 * 224) // Would contain class indices in real implementation
        };
      }
    };
    
    // Create mock image data
    console.log('Creating mock image data...');
    const mockImage = {
      width: 224,
      height: 224,
      channels: 3,
      data: new Uint8Array(3 * 224 * 224).fill(128) // Gray image
    };
    
    // Run the pipeline
    console.log('Running computer vision pipeline...');
    
    // Step 1: Classification
    console.log('Step 1: Classification');
    const classificationResult = await pipeline.classify(mockImage);
    console.log('Classification result:', classificationResult);
    
    // Step 2: Detection
    console.log('Step 2: Object Detection');
    const detectionResult = await pipeline.detect(mockImage);
    console.log(`Detected ${detectionResult.length} objects:`);
    detectionResult.forEach(detection => {
      console.log(`- ${detection.class} (${(detection.confidence * 100).toFixed(1)}%)`);
    });
    
    // Step 3: Segmentation
    console.log('Step 3: Segmentation');
    const segmentationResult = await pipeline.segment(mockImage);
    console.log('Segmentation complete:');
    console.log(`- Image size: ${segmentationResult.width}x${segmentationResult.height}`);
    console.log(`- Classes: ${segmentationResult.classes.join(', ')}`);
    
    return {
      classification: classificationResult,
      detection: detectionResult,
      segmentation: segmentationResult
    };
  } catch (error) {
    console.error('Error in computer vision pipeline:', error);
    throw error;
  }
}

// Run the example
computerVisionPipeline().catch(console.error);
```

## Configuration Options Reference

### Backend Initialization Options

```typescript
interface OVMSUnifiedOptions {
  // Mode settings
  useContainer?: boolean;    // Whether to use container mode (default: false)
  containerUrl?: string;     // URL for the container when in container mode
  
  // API settings
  apiUrl?: string;           // Base URL for the API
  maxRetries?: number;       // Maximum number of retries for failed requests
  requestTimeout?: number;   // Timeout for API requests in milliseconds
  
  // Advanced settings
  circuitBreakerThreshold?: number; // Number of failures before opening circuit
  circuitBreakerResetTime?: number; // Cooldown period in milliseconds
  useRequestQueue?: boolean; // Whether to use the request queue for load balancing
  
  // API key management
  apiKeyMultiplexing?: boolean; // Whether to use multiple API keys
  apiKeys?: ApiKey[];       // List of API keys for multiplexing
  apiKeySelectionStrategy?: 'round-robin' | 'priority' | 'random'; // Selection strategy
  
  // Generation settings (defaults)
  maxTokens?: number;        // Default maximum tokens for generation
  temperature?: number;      // Default temperature for sampling
  topP?: number;             // Default top-p value for sampling
  topK?: number;             // Default top-k value for sampling
  repetitionPenalty?: number; // Default penalty for repeated tokens
  
  // Debugging
  debug?: boolean;           // Whether to enable debug logging
}

interface ApiKey {
  key: string;               // The API key
  priority: 'HIGH' | 'MEDIUM' | 'LOW'; // Key priority
  usage?: number;            // Usage counter (managed internally)
  lastUsed?: number;         // Timestamp of last use
}
```

### Container Deployment Configuration

```typescript
interface DeploymentConfig {
  dockerRegistry: string;    // Docker registry to pull the image from
  containerTag: string;      // Container tag to use
  gpuDevice: string;         // GPU device to use ('0', '1', etc., or '' for CPU)
  modelId: string;           // Model ID to deploy
  port: number;              // Port to expose
  env?: Record<string, string>; // Environment variables
  volumes?: string[];         // Volume mounts
  network?: string;           // Docker network to use
  maxInputLength?: number;    // Maximum input length for the model
  parameters?: string[];      // Additional parameters for the container
}
```

### Model Configuration

```typescript
interface OVMSModelConfig {
  batch_size?: number;       // Maximum batch size
  preferred_batch?: number;  // Preferred batch size for optimal performance
  instance_count?: number;   // Number of model instances to run in parallel
  execution_mode?: 'latency' | 'throughput'; // Optimization priority
  [key: string]: any;        // Additional model-specific configuration
}
```

### Quantization Configuration

```typescript
interface OVMSQuantizationConfig {
  enabled: boolean;          // Whether quantization is enabled
  method?: string;           // Quantization method (e.g., "MinMax", "KL")
  bits?: number;             // Bit width for quantization (8, 4, etc.)
  [key: string]: any;        // Additional quantization parameters
}
```

## Advanced Hardware Optimization Tips

OpenVINO Model Server is optimized for Intel hardware and provides several ways to maximize performance:

### CPU Optimization

For Intel CPUs:
- Enable `execution_mode: 'throughput'` for multi-core workloads
- Set `instance_count` to match available CPU cores
- Use INT8 quantization for 2-4x speedup with minimal accuracy loss

### GPU Optimization

For Intel GPUs:
- Set `gpuDevice` to your Intel GPU device ID
- Use FP16 precision for best GPU performance
- Balance `batch_size` according to available GPU memory

### Neural Compute Stick 2 (NCS2) Optimization

For Intel Neural Compute Stick 2:
- Use small-medium sized models that fit on the device
- Prefer INT8 quantization for best performance
- Limit batch size to 1 or 2 for optimal processing

### Multi-Device Configuration

For systems with multiple accelerators:
- Distribute models across available hardware
- Configure separate containers for CPU, GPU, and specialized hardware
- Use API key multiplexing to route requests to the most appropriate backend

## Interface Reference

### Methods

| Method | Description |
|--------|-------------|
| `constructor(options, metadata)` | Initializes the backend with the given options and metadata. |
| `setMode(containerMode)` | Sets the mode to API or container. |
| `getMode()` | Gets the current mode ('api' or 'container'). |
| `testEndpoint()` | Tests if the endpoint is available. |
| `infer(model, data, options)` | Runs inference with the specified model and data. |
| `batchInfer(model, dataBatch, options)` | Runs batch inference with the specified model and data batch. |
| `inferWithVersion(model, version, data, options)` | Runs inference with a specific model version. |
| `getModelInfo(model)` | Gets detailed information about a model. |
| `getModelVersions(model)` | Gets available versions for a model. |
| `getModelStatus(model)` | Gets the status of a model. |
| `setModelConfig(model, config)` | Sets configuration for a model. |
| `setExecutionMode(model, mode)` | Sets execution mode for a model. |
| `setQuantization(model, config)` | Sets quantization for a model. |
| `reloadModel(model)` | Reloads a model. |
| `getServerStatistics()` | Gets server statistics. |
| `getModelMetadataWithShapes(model)` | Gets model metadata with input/output shapes. |
| `startContainer(config)` | Starts a container with the specified configuration. |
| `stopContainer()` | Stops the current container. |
| `createEndpointHandler(endpointUrl, model)` | Creates a handler for a specific endpoint. |
| `formatRequest(handler, input)` | Formats input data and sends request using the provided handler. |
| `explainPrediction(model, data, options)` | Gets explanation for a prediction. |
| `runBenchmark(options)` | Runs a performance benchmark. |

### Common Response Types

#### Model Metadata

```typescript
interface OVMSModelMetadata {
  name: string;               // Model name
  versions: string[];         // Available model versions
  platform: string;           // Model platform (e.g., "openvino", "tensorflow")
  inputs: OVMSModelInput[];   // Model input specifications
  outputs: OVMSModelOutput[]; // Model output specifications
  [key: string]: any;         // Additional metadata fields
}

interface OVMSModelInput {
  name: string;               // Input tensor name
  datatype: string;           // Data type (e.g., "FP32", "INT8")
  shape: number[];            // Input shape dimensions
  layout?: string;            // Optional tensor layout (e.g., "NHWC", "NCHW")
  [key: string]: any;         // Additional input specifications
}

interface OVMSModelOutput {
  name: string;               // Output tensor name
  datatype: string;           // Data type (e.g., "FP32", "INT8")
  shape: number[];            // Output shape dimensions
  layout?: string;            // Optional tensor layout
  [key: string]: any;         // Additional output specifications
}
```

#### Server Statistics

```typescript
interface OVMSServerStatistics {
  server_uptime?: number;     // Server uptime in seconds
  server_version?: string;    // OVMS server version
  active_models?: number;     // Number of active models
  total_requests?: number;    // Total processed requests
  requests_per_second?: number; // Request throughput
  avg_inference_time?: number; // Average inference time in milliseconds
  cpu_usage?: number;         // CPU usage percentage
  memory_usage?: number;      // Memory usage in MB
  [key: string]: any;         // Additional statistics
}
```

#### Container Information

```typescript
interface ContainerInfo {
  containerId: string;        // Docker container ID
  host: string;               // Host address (usually localhost)
  port: number;               // Exposed port
  status: string;             // Container status
  model: string;              // Model ID deployed in the container
  startTime: number;          // Container start timestamp
}
```

## OpenVINO Integration

The OVMS Unified backend is optimized for working with models that have been converted to OpenVINO format using the Model Optimizer tool. This enables high-performance inference on Intel hardware.

### Key OpenVINO Features

1. **Cross-Platform Performance**: Optimized performance on Intel CPUs, GPUs, VPUs, and FPGAs
2. **Quantization Support**: INT8 and INT4 quantization for improved performance and reduced memory footprint
3. **Automatic Hardware Selection**: Models can run on the most appropriate available hardware
4. **Model Optimization**: Built-in optimizations for specific hardware targets
5. **Wide Model Support**: Compatible with TensorFlow, PyTorch, ONNX, MXNet, Caffe, and more

### Supported Model Types

The OpenVINO Model Server supports a wide range of model types:

- **Computer Vision**: Image classification, object detection, semantic segmentation, instance segmentation, pose estimation
- **Natural Language Processing**: Text classification, named entity recognition, sentiment analysis, machine translation
- **Audio Processing**: Speech recognition, voice activity detection, speaker identification
- **Recommendation Systems**: User-item models, collaborative filtering
- **Anomaly Detection**: Time series, image, and text anomaly detection

For more information on optimizing models for OpenVINO, see the [OpenVINO documentation](https://docs.openvino.ai/).

## Best Practices

### Performance Optimization

1. **Choose the right batch size**: Larger batch sizes typically provide better throughput, but may increase latency for individual requests. Use batch processing when processing multiple inputs simultaneously.

2. **Use quantization**: Enable INT8 quantization for significant performance improvements with minimal impact on accuracy.

3. **Optimize for your use case**: Use `execution_mode: 'latency'` for real-time applications and `execution_mode: 'throughput'` for batch processing.

4. **Set appropriate instance count**: Generally, set `instance_count` to match the number of available CPU cores or GPU compute units.

5. **Use container mode for high-throughput applications**: Container mode avoids API overhead and provides more control over model deployment.

### Resource Management

1. **Monitor resource usage**: Keep track of CPU, memory, and GPU utilization to ensure optimal performance.

2. **Stop containers when not in use**: Free up resources by stopping containers that are not actively processing requests.

3. **Scale horizontally for high demand**: Deploy multiple containers on different ports or machines and use load balancing.

### Error Handling

1. **Implement robust error handling**: Use the retry mechanism with exponential backoff for transient errors.

2. **Set appropriate timeouts**: Configure request timeouts based on your model's processing time requirements.

3. **Use circuit breaker pattern**: Automatically disable requests when the server is experiencing issues to prevent cascading failures.

## Security Considerations

1. **API key protection**: Store API keys securely and never expose them in client-side code.

2. **Container security**: If exposing container endpoints publicly, implement proper authentication and authorization.

3. **Input validation**: Always validate and sanitize inputs before passing them to the model to prevent injection attacks.

4. **Network security**: Consider using HTTPS and restricting access to model server endpoints.

## Additional Resources

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO Model Server GitHub Repository](https://github.com/openvinotoolkit/model_server)
- [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
- [Intel Developer Zone - OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
- [IPFS Accelerate JS Documentation](../../../README.md)
- [API Backends Development Guide](./API_BACKEND_DEVELOPMENT.md)
- [API Backend Interface Reference](./API_BACKEND_INTERFACE_REFERENCE.md)