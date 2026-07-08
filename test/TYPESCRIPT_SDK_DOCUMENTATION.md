# IPFS Accelerate JavaScript SDK Documentation

**Current Version:** 1.0.0  
**Updated:** March 18, 2025  
**Status:** Ready for Release (100% complete)

## Overview

The IPFS Accelerate JavaScript SDK provides optimized machine learning inference capabilities in the browser using WebNN and WebGPU hardware acceleration. It enables efficient AI model execution with features such as cross-model tensor sharing, model weight caching, and adaptive hardware selection.

## Key Features

- **Hardware Acceleration**: Leverages WebNN and WebGPU for optimal performance across browsers
- **Model Compatibility**: Supports popular model architectures like ViT, BERT, Whisper, and more
- **Model Weight Caching**: Persistent storage of model weights using IndexedDB
- **Adaptive Hardware Selection**: Automatically selects the best available hardware backend
- **Cross-Browser Optimization**: Specialized optimizations for Chrome, Firefox, Edge, and Safari
- **Tensor Operations**: Comprehensive suite of tensor operations with hardware acceleration
- **Memory Efficiency**: Intelligent memory management with garbage collection
- **API Backends**: Comprehensive TypeScript implementations of API clients for major AI providers
- **Cross-Model Tensor Sharing**: Efficiently share tensors between models for memory optimization
- **Ultra-Low Precision**: 2-bit, 3-bit, and 4-bit quantization support for memory-efficient inference
- **Container Management**: Built-in Docker container management for local AI model servers

## Installation

```bash
npm install ipfs-accelerate
```

For development from source:

```bash
git clone https://github.com/ipfs-accelerate/ipfs-accelerate-js.git
cd ipfs-accelerate-js
npm install
npm run build
```

## Quick Start Example

```javascript
import { WebNNBackend, WebNNStorageIntegration } from 'ipfs-accelerate';

async function runInference() {
  // Create and initialize WebNN backend
  const backend = new WebNNBackend();
  await backend.initialize();
  
  // Create storage integration
  const storage = new WebNNStorageIntegration(backend);
  await storage.initialize();
  
  // Check if model is already cached
  const modelId = 'bert-base-uncased';
  if (await storage.isModelCached(modelId)) {
    // Load model from cache
    const modelTensors = await storage.loadModel(modelId);
    
    // Create input tensor
    const inputIds = new Int32Array([101, 2054, 2003, 2026, 2171, 1029, 102]);
    const inputTensor = await backend.createTensor(
      inputIds,
      [1, 7],
      'int32'
    );
    
    // Run inference
    const result = await backend.execute('bert', {
      input_ids: inputTensor,
      weights: modelTensors
    });
    
    // Get output
    const outputData = await backend.readTensor(result.tensor, result.shape);
    console.log('Inference result:', outputData);
  } else {
    console.log('Model not cached. Please download and store it first.');
  }
}

runInference();
```

## Architecture

The SDK follows a modular architecture with several key components:

### Hardware Abstraction Layer (HAL)

Provides a consistent interface for different hardware backends:

```typescript
interface HardwareBackend {
  type: HardwareBackendType;
  isSupported(): Promise<boolean>;
  initialize(): Promise<boolean>;
  getCapabilities(): Promise<Record<string, any>>;
  createTensor(data: TypedArray, shape: number[], dataType?: string): Promise<any>;
  execute(operation: string, inputs: Record<string, any>, options?: Record<string, any>): Promise<any>;
  readTensor(tensor: any, shape: number[], dataType?: string): Promise<TypedArray>;
  dispose(): void;
}
```

### WebNN Backend

Implements the Hardware Abstraction Layer using the WebNN API for neural network acceleration:

```typescript
const backend = new WebNNBackend({
  deviceType: 'gpu',               // 'gpu' or 'cpu'
  powerPreference: 'high-performance', // 'high-performance', 'low-power', or 'default'
  enableLogging: true,             // Enable logging for debugging
  floatPrecision: 'float32',       // 'float32' or 'float16'
  preferSyncExecution: true,       // Use sync execution when available
  memory: {                       
    enableGarbageCollection: true, // Enable garbage collection of unused tensors
    garbageCollectionThreshold: 1024 * 1024 * 128 // 128MB threshold
  }
});

await backend.initialize();
```

### WebNN Standalone Interface

Provides an easier-to-use interface for WebNN capabilities without requiring the full HAL:

```typescript
import { 
  isWebNNSupported, 
  getWebNNDeviceInfo, 
  getWebNNBrowserRecommendations,
  runWebNNExample
} from 'ipfs-accelerate/webnn-standalone';

// Check if WebNN is supported
const supported = await isWebNNSupported();

// Get device information
const deviceInfo = await getWebNNDeviceInfo();

// Get browser recommendations
const recommendations = getWebNNBrowserRecommendations();

// Run a simple example operation
const result = await runWebNNExample('matmul');
```

### Storage Manager

Provides persistent storage for model weights and tensors using IndexedDB:

```typescript
const storageManager = new StorageManager({
  dbName: 'my-model-cache',        // Name of the IndexedDB database
  enableCompression: true,         // Enable compression for stored tensors
  maxStorageSize: 1024 * 1024 * 1024, // 1GB maximum storage size
  enableAutoCleanup: true,         // Enable automatic cleanup of unused items
  cleanupThreshold: 1000 * 60 * 60 * 24 * 7, // 7 days threshold
  enableLogging: true              // Enable logging for debugging
});

await storageManager.initialize();
```

### WebNN Storage Integration

Integrates the WebNN backend with the Storage Manager for efficient model loading:

```typescript
const storage = new WebNNStorageIntegration(backend, {
  enableModelCaching: true,        // Enable model weights caching
  enableAutoCleanup: true,         // Enable automatic cleanup of unused models
  maxStorageSize: 1024 * 1024 * 1024, // 1GB maximum storage size
  dbName: 'webnn-model-cache'      // Database name
});

await storage.initialize();

// Store a model
await storage.storeModel(
  'my-model',                      // Model ID
  'My Model',                      // Model name
  tensorMap,                       // Map of tensor names to tensor data
  { version: '1.0.0' }             // Optional metadata
);

// Load a model
const modelTensors = await storage.loadModel('my-model');
```

## Supported Operations

The SDK supports the following neural network operations with hardware acceleration:

### Core Operations
- **Matrix Multiplication**: Multiply two matrices
- **Convolution**: 2D convolution for convolutional neural networks
- **Softmax**: Softmax activation function
- **Elementwise**: Basic elementwise operations (relu, sigmoid, tanh)

### Pooling Operations
- **Max Pooling**: Maximum value in a sliding window
- **Average Pooling**: Average value in a sliding window

### Normalization Operations
- **Batch Normalization**: Normalize across the batch dimension
- **Layer Normalization**: Normalize across specified axes

### Advanced Elementwise Operations
- **Add**: Element-wise addition
- **Subtract**: Element-wise subtraction
- **Multiply**: Element-wise multiplication
- **Divide**: Element-wise division
- **Power**: Element-wise exponentiation
- **Minimum**: Element-wise minimum
- **Maximum**: Element-wise maximum
- **Exponential**: Element-wise exponential function
- **Logarithm**: Element-wise natural logarithm
- **Square Root**: Element-wise square root

### Tensor Manipulation Operations
- **Reshape**: Change tensor shape without altering data
- **Transpose**: Permute tensor dimensions
- **Concatenate**: Join tensors along a specified axis
- **Slice**: Extract a slice from a tensor
- **Pad**: Pad a tensor with a constant value

## Browser Compatibility

| Browser | WebNN Support | WebGPU Support | Recommended For |
|---------|---------------|----------------|-----------------|
| Edge    | Excellent     | Good           | WebNN operations, general use |
| Chrome  | Good          | Excellent      | WebGPU operations, vision models |
| Firefox | Limited       | Good           | Audio models, compute shaders |
| Safari  | Experimental  | Good           | Vision models on Apple Silicon |

## Performance Considerations

### Device Selection
Choose the appropriate device type for your workload:
```typescript
const backend = new WebNNBackend({
  deviceType: 'gpu', // Use GPU for most neural network operations
  powerPreference: 'high-performance' // Prioritize performance over battery life
});
```

### Memory Management
Explicitly dispose of tensors when no longer needed:
```typescript
// After you're done with a tensor
backend.dispose();
```

### Model Caching
Use the storage manager to cache models for improved loading times:
```typescript
// Store the model once
await storage.storeModel('model-id', 'Model Name', tensorMap);

// Then load it efficiently in future sessions
const modelTensors = await storage.loadModel('model-id');
```

## Error Handling

The SDK uses async/await patterns for most operations, so use try/catch for error handling:

```javascript
try {
  const result = await backend.execute('matmul', {
    a: inputTensor,
    b: weightsTensor
  });
  
  const output = await backend.readTensor(result.tensor, result.shape);
  console.log('Success:', output);
} catch (error) {
  console.error('Inference failed:', error);
  
  // Handle specific error types
  if (error.name === 'NotSupportedError') {
    console.error('This operation is not supported on your device');
  } else if (error.name === 'OutOfMemoryError') {
    console.error('Not enough memory to run this model');
  }
}
```

## Examples

### Basic Tensor Operations

```javascript
// Create input tensors
const tensorA = await backend.createTensor(
  new Float32Array([1, 2, 3, 4]),
  [2, 2],
  'float32'
);

const tensorB = await backend.createTensor(
  new Float32Array([5, 6, 7, 8]),
  [2, 2],
  'float32'
);

// Matrix multiplication
const result = await backend.execute('matmul', {
  a: tensorA,
  b: tensorB
});

// Read the result
const output = await backend.readTensor(result.tensor, result.shape);
console.log('Result:', Array.from(output));
```

### Image Classification with ViT

```javascript
// Load the ViT model from storage
const modelTensors = await storage.loadModel('vit-base-patch16-224');

// Preprocess the image
const imageData = await preprocessImage(imageElement, 224, 224);
const inputTensor = await backend.createTensor(
  imageData,
  [1, 224, 224, 3],
  'float32'
);

// Run inference
const result = await backend.execute('vit', {
  input: inputTensor,
  weights: modelTensors
});

// Get class predictions
const logits = await backend.readTensor(result.tensor, result.shape);
const predictions = getTopClasses(logits, classLabels, 5);
console.log('Top predictions:', predictions);
```

### Text Embedding with BERT

```javascript
// Load BERT model from storage
const modelTensors = await storage.loadModel('bert-base-uncased');

// Tokenize input text
const tokens = tokenizer.encode('Hello, world!');
const inputTensor = await backend.createTensor(
  new Int32Array(tokens),
  [1, tokens.length],
  'int32'
);

// Run inference
const result = await backend.execute('bert', {
  input_ids: inputTensor,
  weights: modelTensors
});

// Get embedding
const embedding = await backend.readTensor(result.tensor, result.shape);
console.log('Text embedding:', embedding);
```

## API Backends

The SDK includes comprehensive TypeScript implementations of API clients for major AI model providers. These backends provide consistent interfaces for interacting with various AI services, whether cloud-based or self-hosted:

### Common Features Across All API Backends

- **Unified Interface**: Consistent method naming and parameter structures
- **Robust Error Handling**: Detailed error types with proper categorization
- **Circuit Breaker Pattern**: Prevents cascading failures by temporarily disabling endpoints after consecutive errors
- **Request Queue Management**: Handles rate limiting and prioritizes requests
- **Streaming Support**: Real-time token generation with AsyncGenerator pattern
- **Retry Mechanism**: Exponential backoff for transient errors
- **API Key Management**: Secure handling of API credentials with environment variable fallbacks
- **Configurable Timeouts**: Custom timeout settings for different operations
- **Request Tracing**: Optional request IDs for tracing and debugging
- **Performance Metrics**: Track token usage, latency, and throughput

### OpenAI Backend

```typescript
import { OpenAI } from 'ipfs-accelerate/api_backends/openai';

// Initialize the backend
const openai = new OpenAI({}, {
  api_key: process.env.OPENAI_API_KEY
});

// Generate text
const completion = await openai.generateText("Write a poem about AI", {
  model: "gpt-4o",
  temperature: 0.7,
  maxTokens: 150
});

console.log(completion);
```

### HuggingFace Text Generation Inference (TGI) Unified Backend

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';

// Initialize with API or container mode
const hfTgi = new HfTgiUnified({
  useContainer: false // Set to true for container mode
}, {
  hf_api_key: process.env.HF_API_KEY,
  model_id: 'meta-llama/Llama-2-7b-chat-hf'
});

// Generate text
const response = await hfTgi.generateText(
  "Explain quantum computing in simple terms", 
  { maxTokens: 100 }
);

console.log(response);

// Stream chat responses
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is machine learning?' }
];

for await (const chunk of hfTgi.streamChat(messages)) {
  process.stdout.write(chunk.content);
}
```

### VLLM Unified Backend

```typescript
import { VllmUnified } from 'ipfs-accelerate/api_backends/vllm_unified';

// Initialize the backend
const vllm = new VllmUnified({}, {
  vllm_api_url: 'http://localhost:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
});

// Basic text generation
const result = await vllm.makeRequest(
  'http://localhost:8000',
  'Tell me a story about dragons',
  'meta-llama/Llama-2-7b-chat-hf'
);

console.log(result.text);

// Process batch requests
const prompts = [
  "What is the capital of France?",
  "What is the capital of Germany?"
];

const batchResults = await vllm.processBatch(
  'http://localhost:8000',
  prompts,
  'meta-llama/Llama-2-7b-chat-hf'
);

batchResults.forEach((result, i) => console.log(`Q: ${prompts[i]}\nA: ${result}`));
```

### OVMS (OpenVINO Model Server) Unified Backend

```typescript
import { OvmsUnified } from 'ipfs-accelerate/api_backends/ovms_unified';

// Initialize the backend
const ovms = new OvmsUnified({}, {
  ovms_api_url: 'http://localhost:9000',
  model_name: 'bert-base-uncased'
});

// Run inference on a vision model
const imageBuffer = fs.readFileSync('image.jpg');
const encodedImage = imageBuffer.toString('base64');

const result = await ovms.classifyImage(
  encodedImage,
  {
    model: 'resnet50',
    topK: 5,
    threshold: 0.1
  }
);

console.log('Image classifications:', result);
```

### Other Supported API Backends

The SDK includes implementations for all major AI providers:

- **Groq**: High-performance LLM API with ultra-low latency
- **Claude**: Anthropic's Claude API with advanced context capabilities
- **Gemini**: Google's Gemini API with multimodal capabilities
- **Ollama**: Local LLM deployment for self-hosted models
- **HF TEI Unified**: HuggingFace Text Embedding Inference with both API and container modes
- **HF TGI Container**: Docker container management for HuggingFace models
- **LLVM**: Low-level virtual machine for optimized inference
- **VLLM**: High-throughput serving for LLMs with tensor parallelism
- **OVMS**: OpenVINO Model Server for optimized Intel hardware inference
- **S3 Kit**: S3-compatible model hosting and management

### API Key Management Best Practices

For secure management of API keys and credentials, the SDK supports multiple approaches:

```typescript
// Environment variables (recommended for production)
// Set these in your environment or .env file (use dotenv package)
process.env.OPENAI_API_KEY = "your-api-key";
process.env.HF_API_KEY = "your-huggingface-key";
process.env.GROQ_API_KEY = "your-groq-key";

// Directly in code (only for development)
const openai = new OpenAI({}, { 
  api_key: "your-openai-key" 
});

// Using API key rotation
const openai = new OpenAI({}, {});
openai.createEndpoint({
  apiKey: "key1",
  maxConcurrentRequests: 10,
  priority: "HIGH"
});
openai.createEndpoint({
  apiKey: "key2",
  maxConcurrentRequests: 5,
  priority: "MEDIUM"
});
```

### Error Handling Best Practices

The SDK provides comprehensive error handling with typed errors and detailed information:

```typescript
try {
  const result = await openai.generateText("Sample prompt");
  console.log(result);
} catch (error) {
  // Check error types
  if (error.isRateLimitError) {
    console.log(`Rate limited. Retry after ${error.retryAfter} seconds`);
    setTimeout(() => retry(), error.retryAfter * 1000);
  } else if (error.statusCode === 401) {
    console.log("Authentication error. Check your API key.");
  } else if (error.statusCode === 404) {
    console.log("Model not found. Check model name.");
  } else if (error.isTransientError) {
    console.log("Temporary server error. Safe to retry.");
  } else {
    console.log(`Error: ${error.message}`);
  }
}
```

### The Unified Interface Pattern

All API backends implement a consistent interface pattern for both basic and advanced functionality:

```typescript
// Common interface methods for all backends
interface ApiBackend {
  // Basic text generation
  generateText(prompt: string, options?: any): Promise<string>;
  
  // Chat-based interaction
  chat(messages: Message[], options?: any): Promise<ChatResponse>;
  
  // Streaming responses
  streamGenerateText(prompt: string, options?: any): AsyncGenerator<StreamChunk>;
  streamChat(messages: Message[], options?: any): AsyncGenerator<StreamChunk>;
  
  // API connection testing
  testEndpoint(): Promise<boolean>;
  
  // Endpoint management
  createEndpoint(options: EndpointOptions): string;
  getEndpointStats(endpointId: string): EndpointStats;
}
```

This unified interface allows for easy switching between different providers without changing your application code:

```typescript
// Initialize with one backend
let backend: ApiBackend = new OpenAI({}, { api_key: process.env.OPENAI_API_KEY });

// Switch to another backend if needed
if (needLowerLatency) {
  backend = new Groq({}, { api_key: process.env.GROQ_API_KEY });
}

// Use the same interface regardless of backend
const result = await backend.generateText("Write a poem about AI");
```

### Container Management

The "Unified" backend variants (HF TGI Unified, HF TEI Unified, OVMS Unified, VLLM Unified) include built-in container management for self-hosted deployments:

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';

// Create a backend in container mode
const tgi = new HfTgiUnified({
  useContainer: true, // Enable container mode
}, {
  model_id: 'meta-llama/Llama-2-7b-chat-hf'
});

// Define container configuration
const config = {
  dockerRegistry: 'ghcr.io/huggingface/text-generation-inference',
  containerTag: 'latest',
  gpuDevice: '0', // Use first GPU, use empty string for CPU-only
  modelId: 'meta-llama/Llama-2-7b-chat-hf',
  port: 8080,
  volumes: ['./cache:/data'], // Mount local cache directory
  env: {
    'HF_API_TOKEN': process.env.HF_API_KEY // For private models
  }
};

// Start the container
const containerInfo = await tgi.startContainer(config);
console.log(`Container running at ${containerInfo.host}:${containerInfo.port}`);

// Use the container normally
const result = await tgi.generateText("Hello, world!");

// Stop the container when done
await tgi.stopContainer();
```

The container management features include:

- **Automated Container Lifecycle**: Start, stop, pause, and resume containers
- **Resource Management**: Control GPU allocation, memory limits, and CPU allocation
- **Volume Mounting**: Mount local directories for caching and persistence
- **Environment Variables**: Configure container behavior with environment variables
- **Health Monitoring**: Track container health and resource usage
- **Port Management**: Configure port mapping for container access
- **Multi-Container Orchestration**: Manage multiple containers for high availability

## Current Status and Roadmap

The SDK is now ready for release with 100% of planned features implemented. The following components are complete:

- ✅ WebNN backend implementation
- ✅ WebNN standalone interface 
- ✅ Additional WebNN operations (pooling, normalization, elementwise, manipulation)
- ✅ Storage Manager for model weights with IndexedDB
- ✅ WebNN Storage Integration
- ✅ Operation fusion for better performance
- ✅ WebGPU compute shader operations 
- ✅ WGSL shader implementations for core tensor operations
- ✅ Cross-model tensor sharing with reference counting
- ✅ Browser-specific shader optimizations
- ✅ Model implementations (ViT, BERT, Whisper)
- ✅ API Backends for major AI providers
- ✅ Comprehensive examples for all backends and model types
- ✅ NPM package preparation for release

Future enhancements (planned for Q3-Q4 2025):

- 🔲 Enhanced mobile device support
- 🔲 Advanced model compression techniques
- 🔲 Federated learning capabilities
- 🔲 Extended support for vision transformers
- 🔲 Advanced memory optimization techniques

## Advanced Usage Examples

### Multi-Backend Orchestration

Combine multiple backends for resilience and optimal performance:

```typescript
import { OpenAI } from 'ipfs-accelerate/api_backends/openai';
import { Groq } from 'ipfs-accelerate/api_backends/groq';
import { Claude } from 'ipfs-accelerate/api_backends/claude';
import { MultiBackendOrchestrator } from 'ipfs-accelerate/api_backends/orchestration';

// Create multiple backends
const openai = new OpenAI({}, { api_key: process.env.OPENAI_API_KEY });
const groq = new Groq({}, { api_key: process.env.GROQ_API_KEY });
const claude = new Claude({}, { api_key: process.env.CLAUDE_API_KEY });

// Create orchestrator with prioritized backends
const orchestrator = new MultiBackendOrchestrator([
  { backend: groq, priority: 10, weight: 40 },  // Prioritize Groq for speed
  { backend: openai, priority: 8, weight: 40 }, // Use OpenAI as secondary
  { backend: claude, priority: 5, weight: 20 }  // Use Claude as fallback
]);

// Route requests optimally based on type, latency, and availability
const result = await orchestrator.generateText("Tell me about quantum computing");

// Stream with automatic fallback if primary fails
const stream = orchestrator.streamChat([
  { role: 'user', content: 'Explain neural networks' }
]);

for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

### Hardware + API Integration

Combine WebGPU hardware acceleration with API backends for optimal processing:

```typescript
import { WebGPUBackend } from 'ipfs-accelerate/hardware/webgpu';
import { OpenAI } from 'ipfs-accelerate/api_backends/openai';
import { TensorUtils } from 'ipfs-accelerate/tensor/utils';

// Initialize hardware backend for local processing
const hardware = new WebGPUBackend();
await hardware.initialize();

// Initialize API backend for remote processing
const api = new OpenAI({}, { api_key: process.env.OPENAI_API_KEY });

// Example: Image processing with local hardware + API integration
async function processImageAndDescribe(imageUrl) {
  // Load and preprocess image locally using WebGPU
  const imageData = await loadImage(imageUrl);
  
  // Create tensor from image data
  const inputTensor = await hardware.createTensor(
    new Float32Array(imageData),
    [1, 224, 224, 3]
  );
  
  // Run local feature extraction with WebGPU
  const featuresResult = await hardware.execute('vit_features', {
    input: inputTensor
  });
  
  // Extract features from result
  const features = await hardware.readTensor(featuresResult.tensor);
  
  // Convert features to embedding array
  const embedding = Array.from(features);
  
  // Send embedding to API for description
  const description = await api.generateText({
    prompt: "Describe this image based on its embedding:",
    embedding: embedding.slice(0, 1024) // Send first 1024 features
  });
  
  return description;
}

// Process an image
const description = await processImageAndDescribe("https://example.com/image.jpg");
console.log("Image description:", description);
```

### Cross-Browser Model Sharding

Run large models across multiple browser tabs for better performance:

```typescript
import { CrossBrowserModelSharding } from 'ipfs-accelerate/browser/model_sharding';

// Create a model sharding manager
const sharding = new CrossBrowserModelSharding({
  model: 'llama-70b',  // Large model to shard
  shardType: 'layer',  // Shard by model layers
  numShards: 4,        // Use 4 browser tabs
  browsers: {
    chrome: { weight: 2 },  // Use Chrome for 2 shards
    firefox: { weight: 1 }, // Use Firefox for 1 shard
    edge: { weight: 1 }     // Use Edge for 1 shard
  }
});

// Initialize sharding (opens browser tabs)
await sharding.initialize();

// Run large model inference across multiple browsers
const result = await sharding.generateText(
  "Explain the theory of relativity in detail", 
  { maxLength: 2000 }
);

// Clean up when done (closes browser tabs)
await sharding.cleanup();
```

## NPM Package Publishing

To publish the package to NPM:

```bash
# Build the package
npm run build:full

# Run tests to verify everything works
npm test

# Generate documentation
npm run docs

# Publish to NPM
npm publish
```

### Package Structure

The published NPM package has the following structure:

```
ipfs-accelerate/
├── dist/                    # Compiled JavaScript files
│   ├── index.js              # CommonJS bundle
│   ├── index.esm.js          # ES modules bundle 
│   ├── ipfs-accelerate.min.js # Minified UMD bundle
│   └── types/                # TypeScript declarations
├── src/                     # Source TypeScript files
│   ├── api_backends/         # API client implementations
│   │   ├── openai/           # OpenAI API client
│   │   ├── hf_tgi_unified/   # HuggingFace TGI client
│   │   ├── vllm_unified/     # VLLM client
│   │   ├── ovms_unified/     # OpenVINO Model Server client
│   │   └── ...               # Other API clients
│   ├── hardware/             # Hardware abstraction layer
│   ├── storage/              # Storage implementations
│   ├── tensor/               # Tensor operations
│   └── utils/                # Utility functions
├── docs/                    # Generated documentation
└── examples/                # Usage examples
```

## Contributing

Contributions to this SDK are welcome! Here's how to get started:

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes, adding tests for new functionality
5. Run tests to ensure everything works correctly
6. Submit a pull request to the main repository

Please follow the project's code style and include appropriate tests with your contributions.

## License

This SDK is released under the GNU Affero General Public License v3.0 (AGPL-3.0). See the LICENSE file for the complete text of the license.

## Acknowledgments

This SDK builds upon the work of numerous open-source projects and standards, including:

- **Web API Standards**:
  - WebNN API specification
  - WebGPU API specification
  - WebAssembly (WASM)
  - Web Storage API

- **AI Frameworks and Libraries**:
  - TensorFlow.js
  - ONNX Runtime Web
  - Transformers.js
  - LiteLLM

- **Model Serving Technologies**:
  - HuggingFace Text Generation Inference (TGI)
  - HuggingFace Text Embedding Inference (TEI)
  - OpenVINO Model Server (OVMS)
  - VLLM
  - Ollama

- **Web Technologies**:
  - TypeScript
  - Web Workers
  - IndexedDB
  - WebSockets
  - Service Workers

- **Infrastructure**:
  - Docker
  - REST API standards
  - gRPC

Special thanks to all the contributors and the open-source community for their valuable work which made this SDK possible.