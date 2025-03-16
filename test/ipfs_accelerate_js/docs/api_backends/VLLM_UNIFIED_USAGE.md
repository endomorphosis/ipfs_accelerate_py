# VLLM Unified API Backend Usage Guide

The VLLM Unified API backend provides comprehensive integration with VLLM servers, enabling advanced text generation capabilities, batched inference, streaming responses, model management, and specialized features like LoRA adapters and quantization settings. This unified backend is part of the ongoing TypeScript migration effort and offers enhanced functionality over the standard VLLM backend.

## Key Features

- **Standard Text Generation**: Simple API for text completion and chat-based generation
- **Batch Processing**: Efficient handling of multiple prompts in a single request
- **Streaming Generation**: Real-time token-by-token streaming for responsive UIs
- **Model Management**: API for querying model information and statistics
- **LoRA Adapters**: Support for listing and loading LoRA adapters
- **Quantization Control**: API for configuring model quantization settings
- **Comprehensive Error Handling**: Graceful handling of rate limits and server errors
- **Resource Pooling**: Efficient connection management and request queueing

## Basic Usage

### Initialization

```typescript
import { VllmUnified } from 'ipfs_accelerate_js/api_backends/vllm_unified';

// Simple initialization with defaults
const vllm = new VllmUnified();

// With custom settings
const vllm = new VllmUnified(
  {}, // resources (optional)
  {
    vllm_api_url: 'http://your-vllm-server:8000',
    vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
    timeout: 60000 // milliseconds
  }
);
```

### Text Generation

```typescript
// Basic generation with a string prompt
const result = await vllm.makeRequest(
  'http://your-vllm-server:8000',
  'Tell me a short story about dragons',
  'llama-7b'
);
console.log(result.text);

// Generation with parameters
const resultWithParams = await vllm.makeRequest(
  'http://your-vllm-server:8000',
  {
    prompt: 'Tell me a short story about dragons',
    max_tokens: 200,
    temperature: 0.7,
    top_p: 0.95
  },
  'llama-7b'
);
console.log(resultWithParams.text);
```

### Chat Completion

```typescript
// Chat completion with messages
const chatResult = await vllm.chat(
  [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Tell me about dragons.' }
  ],
  {
    model: 'llama-7b-chat',
    temperature: 0.7
  }
);
console.log(chatResult.content);
```

### Streaming Generation

```typescript
// Stream text generation
const stream = vllm.streamGeneration(
  'http://your-vllm-server:8000',
  'Tell me a story about dragons',
  'llama-7b',
  { temperature: 0.7 }
);

for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

### Batch Processing

```typescript
// Process multiple prompts in parallel
const prompts = [
  'What is the capital of France?',
  'What is the capital of Italy?',
  'What is the capital of Germany?'
];

const batchResults = await vllm.processBatch(
  'http://your-vllm-server:8000',
  prompts,
  'llama-7b',
  { temperature: 0.1 }
);

batchResults.forEach((result, index) => {
  console.log(`Question ${index + 1}: ${prompts[index]}`);
  console.log(`Answer: ${result}`);
});
```

## Advanced Features

### Model Information and Statistics

```typescript
// Get model information
const modelInfo = await vllm.getModelInfo(
  'http://your-vllm-server:8000',
  'llama-7b'
);
console.log(`Model max sequence length: ${modelInfo.max_model_len}`);
console.log(`Model dtype: ${modelInfo.dtype}`);
console.log(`GPU memory utilization: ${modelInfo.gpu_memory_utilization * 100}%`);

// Get model statistics
const stats = await vllm.getModelStatistics(
  'http://your-vllm-server:8000',
  'llama-7b'
);
console.log(`Total requests processed: ${stats.statistics.requests_processed}`);
console.log(`Average generation time: ${stats.statistics.avg_generation_time}s`);
console.log(`Throughput: ${stats.statistics.throughput} tokens/second`);
```

### LoRA Adapters Management

```typescript
// List available LoRA adapters
const adapters = await vllm.listLoraAdapters(
  'http://your-vllm-server:8000'
);
console.log(`Available adapters: ${adapters.map(a => a.name).join(', ')}`);

// Load a LoRA adapter
const loadResult = await vllm.loadLoraAdapter(
  'http://your-vllm-server:8000',
  {
    adapter_name: 'MyAdapter',
    adapter_path: '/path/to/adapter',
    base_model: 'llama-7b'
  }
);
console.log(`Adapter loaded: ${loadResult.success}`);
```

### Quantization Control

```typescript
// Configure quantization
const quantizationResult = await vllm.setQuantization(
  'http://your-vllm-server:8000',
  'llama-7b',
  {
    enabled: true,
    method: 'awq',
    bits: 4
  }
);
console.log(`Quantization configuration: ${JSON.stringify(quantizationResult.quantization)}`);
```

## Advanced API Usage

### Custom Endpoint Handlers

```typescript
// Create a custom endpoint handler
const handler = vllm.createVllmEndpointHandler(
  'http://your-vllm-server:8000',
  'llama-7b'
);

// Use the handler
const result = await handler({ prompt: 'Hello, world!' });
console.log(result.text);

// Create a handler with parameters
const paramHandler = vllm.createVllmEndpointHandlerWithParams(
  'http://your-vllm-server:8000',
  'llama-7b',
  { temperature: 0.7, top_p: 0.95 }
);

// The parameters will be automatically applied
const paramResult = await paramHandler({ prompt: 'Hello, world!' });
console.log(paramResult.text);
```

### Endpoint Multiplexing

```typescript
// Create multiple endpoints
const endpoint1 = vllm.createEndpoint({
  api_key: 'key1',
  max_concurrent_requests: 5,
  max_retries: 3
});

const endpoint2 = vllm.createEndpoint({
  api_key: 'key2',
  max_concurrent_requests: 10,
  max_retries: 5
});

// Make requests with specific endpoints
const result1 = await vllm.makeRequestWithEndpoint(
  endpoint1,
  'Hello, world!',
  'llama-7b'
);

const result2 = await vllm.makeRequestWithEndpoint(
  endpoint2,
  'How are you?',
  'llama-7b'
);

// Get endpoint statistics
const stats1 = vllm.getStats(endpoint1);
console.log(`Endpoint 1 requests: ${stats1.requests}`);
console.log(`Endpoint 1 success: ${stats1.success}`);
console.log(`Endpoint 1 errors: ${stats1.errors}`);
```

## Error Handling

The VLLM Unified API backend includes comprehensive error handling:

```typescript
try {
  const result = await vllm.makeRequest(
    'http://your-vllm-server:8000',
    'Tell me a story',
    'non-existent-model'
  );
} catch (error) {
  if (error.statusCode === 404) {
    console.error('Model not found');
  } else if (error.statusCode === 429) {
    console.error(`Rate limited. Retry after ${error.retryAfter} seconds`);
  } else if (error.isTransientError) {
    console.error('Temporary server error, retry later');
  } else {
    console.error(`Error: ${error.message}`);
  }
}
```

## Environment Variables and Configuration

The VLLM Unified API backend can be configured using either metadata during initialization or through environment variables:

### Environment Variables

- `VLLM_API_URL`: Default API URL (e.g., `http://localhost:8000`)
- `VLLM_MODEL`: Default model to use (e.g., `meta-llama/Llama-2-7b-chat-hf`)
- `VLLM_API_KEY`: API key for authentication (if required)
- `VLLM_TIMEOUT`: Request timeout in milliseconds (default: 30000)

### Metadata Configuration Options

When instantiating the API backend, you can provide these settings via metadata:

```typescript
const vllm = new VllmUnified({}, {
  vllm_api_url: 'http://your-vllm-server:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
  vllmApiKey: 'your-api-key',  // Alternative camelCase syntax
  timeout: 60000,  // 60 seconds
  maxRetries: 3,
  maxConcurrentRequests: 10,
  queueSize: 100
});
```

Priority order:
1. Options provided directly in method calls
2. Metadata provided during initialization
3. Environment variables
4. Default values

## Compatibility and Performance

The VLLM Unified API backend is compatible with most transformer-based models and provides high-performance inference through the VLLM server.

### Compatible Model Families

- **Llama Family**: Llama, Llama 2, Llama 3, CodeLlama, etc.
- **Mistral Family**: Mistral, Mixtral models
- **Falcon Family**: Falcon models
- **MPT Models**: Mosaic's MPT models
- **OPT Models**: Meta's OPT series
- **BLOOM Models**: BigScience BLOOM models
- **Other Models**:
  - StableLM models
  - Pythia models
  - Qwen models
  - Claude-compatible models
  - Many other transformer-based models

### Performance and Features

VLLM provides significant performance advantages over standard inference engines:

- **Continuous Batching**: Efficiently handles multiple requests
- **PagedAttention**: Memory-efficient KV cache implementation
- **Tensor Parallelism**: Distributed inference across multiple GPUs
- **Quantization**: Support for various quantization methods (AWQ, SqueezeLLM, etc.)
- **Streaming**: Efficient token-by-token streaming for responsive UI

### Integration with Hardware Backends

The VLLM Unified backend can be easily integrated with hardware backends:

```typescript
import { VllmUnified } from 'ipfs_accelerate_js/api_backends/vllm_unified';
import { HardwareAbstractionLayer } from 'ipfs_accelerate_js/hardware';

// Create VLLM backend
const vllm = new VllmUnified();

// Create hardware abstraction layer
const hal = new HardwareAbstractionLayer();

// Register VLLM as a backend for large language models
hal.registerBackend('llm', vllm);

// Use through the hardware abstraction layer
const result = await hal.runInference('llm', {
  input: 'Hello, world!',
  model: 'llama-7b'
});
```

For a complete list of compatible models and performance features, refer to the [VLLM project documentation](https://github.com/vllm-project/vllm).

## Comparison with Python Implementation

The TypeScript VLLM Unified backend provides all the functionality of the Python implementation with additional features:

| Feature | Python Implementation | TypeScript Implementation |
|---------|----------------------|---------------------------|
| Basic Inference | ✅ | ✅ |
| Chat Completion | ✅ | ✅ |
| Batch Processing | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Queue Management | ✅ | ✅ Enhanced |
| Circuit Breaker | ✅ | ✅ Enhanced |
| Model Information | ✅ | ✅ |
| LoRA Support | ✅ | ✅ |
| Quantization | ✅ | ✅ |
| Performance Tracking | ✅ | ✅ Enhanced |
| Error Handling | ✅ | ✅ Enhanced |
| TypeScript Types | ❌ | ✅ |
| Hardware Integration | ✅ | ✅ Enhanced |

## Contributing and Future Development

The VLLM Unified backend is part of the ongoing TypeScript migration effort. Contributions and feedback are welcome through the project repository.

Future planned enhancements:
- WebSocket-based streaming
- Improved browser integration
- Enhanced batch processing with priority queues
- More comprehensive error reporting
- Extended hardware integration

For more information on contributing, please refer to the project's contribution guidelines.