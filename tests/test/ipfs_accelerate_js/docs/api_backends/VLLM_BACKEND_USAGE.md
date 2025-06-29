# VLLM Backend

The `VLLM` backend provides integration with the VLLM (Very Large Language Model) inference server, a high-throughput and memory-efficient inference engine for LLMs.

This implementation extends the `BaseApiBackend` class, providing a standardized interface that works with the IPFS Accelerate API ecosystem. It can be used interchangeably with other API backends like OpenAI, Groq, Claude, etc., through the unified API interface.

## Features

- Text generation with VLLM-served models
- Streaming text generation
- Chat interface with message formatting
- API key management for secure VLLM deployments
- Multiple endpoint management
- Request tracking and statistics
- Exponential backoff and retries
- Support for batch processing
- Support for LoRA adapters
- Quantization configuration
- Model statistics and information retrieval

## Installation

The VLLM backend is included in the IPFS Accelerate JavaScript SDK. Ensure you have the `ipfs_accelerate_js` package installed.

## Basic Usage

```typescript
import { VLLM } from 'ipfs_accelerate_js/api_backends';

// Initialize with API key and model
const vllm = new VLLM({}, {
  vllm_api_url: 'http://localhost:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
});

// Simple chat completion
async function chatWithModel() {
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain quantum entanglement in simple terms.' }
  ];
  
  const response = await vllm.chat(messages, {
    temperature: 0.7,
    max_tokens: 200
  });
  
  console.log(response.content);
}

// Streaming chat
async function streamingChat() {
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Write a short poem about spring.' }
  ];
  
  for await (const chunk of vllm.streamChat(messages, { temperature: 0.8 })) {
    process.stdout.write(chunk.content);
    if (chunk.done) break;
  }
}
```

## Using with the API Backend Factory

The VLLM backend can be used with the API backend factory for easy creation and model compatibility detection:

```typescript
import { createApiBackend, findCompatibleBackend } from 'ipfs_accelerate_js/api_backends';

// Create by name
const vllm = createApiBackend('vllm', {}, {
  vllm_api_url: 'http://localhost:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
});

// Find compatible backend for a model
const modelName = 'meta-llama/Llama-2-7b-chat-hf';
const backend = findCompatibleBackend(modelName);
if (backend) {
  console.log(`Found compatible backend: ${backend.constructor.name}`);
  const response = await backend.chat([
    { role: 'user', content: 'Hello there!' }
  ]);
}
```

## Advanced Usage

### Custom API Endpoint

```typescript
// For self-hosted VLLM deployments
const customVLLM = new VLLM({}, {
  vllm_api_key: 'your_api_key', // May not be required for local deployments
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
  vllm_api_url: 'http://localhost:8000' // Base URL for your VLLM deployment
});
```

### Endpoint Handlers

```typescript
// Create endpoint handlers for different endpoints
const vllm = new VLLM();

// Create chat endpoint handler
const chatHandler = vllm.createVLLMChatEndpointHandler(
  'http://localhost:8000/v1/chat/completions'
);

// Create completion endpoint handler
const completionHandler = vllm.createEndpointHandler(
  'http://localhost:8000/v1/completions'
);
```

### Batch Processing

```typescript
// Process a batch of prompts
const vllm = new VLLM();
const prompts = [
  "What is artificial intelligence?",
  "Explain machine learning.",
  "What are neural networks?"
];

// Process batch
const results = await vllm.processBatch(
  'http://localhost:8000/v1/completions',
  prompts,
  'meta-llama/Llama-2-7b-chat-hf',
  { temperature: 0.7, max_tokens: 100 }
);

// Process batch with metrics
const [batchResults, metrics] = await vllm.processBatchWithMetrics(
  'http://localhost:8000/v1/completions',
  prompts,
  'meta-llama/Llama-2-7b-chat-hf',
  { temperature: 0.7, max_tokens: 100 }
);

console.log(`Average time per item: ${metrics.average_time_per_item_ms}ms`);
console.log(`Total processing time: ${metrics.total_time_ms}ms`);
```

### Model Information and Statistics

```typescript
// Get model information
const modelInfo = await vllm.getModelInfo('meta-llama/Llama-2-7b-chat-hf');
console.log(`Model max length: ${modelInfo.max_model_len}`);
console.log(`Model GPU memory utilization: ${modelInfo.gpu_memory_utilization}`);

// Get model statistics
const statistics = await vllm.getModelStatistics('meta-llama/Llama-2-7b-chat-hf');
console.log(`Tokens generated: ${statistics.statistics.tokens_generated}`);
console.log(`Average tokens per request: ${statistics.statistics.avg_tokens_per_request}`);
```

### LoRA Adapters

```typescript
// List available LoRA adapters
const adapters = await vllm.listLoraAdapters();
console.log(`Available adapters: ${adapters.map(a => a.name).join(', ')}`);

// Load a LoRA adapter
await vllm.loadLoraAdapter({
  adapter_name: 'my-lora',
  adapter_path: '/path/to/lora',
  base_model: 'meta-llama/Llama-2-7b-chat-hf'
});
```

### Quantization

```typescript
// Set quantization for a model
await vllm.setQuantization('meta-llama/Llama-2-7b-chat-hf', {
  enabled: true,
  method: 'int8',
  bits: 8
});
```

### Testing Endpoints

```typescript
// Test if an endpoint is available
const isAvailable = await vllm.testEndpoint();
console.log(`Endpoint available: ${isAvailable}`);

// Test chat endpoint
const isChatAvailable = await vllm.testVLLMChatEndpoint(
  'http://localhost:8000/v1/chat/completions',
  'meta-llama/Llama-2-7b-chat-hf'
);
console.log(`Chat endpoint available: ${isChatAvailable}`);
```

## API Reference

### Constructor

```typescript
new VLLM(resources?: ApiResources, metadata?: ApiMetadata)
```

**Parameters:**
- `resources`: Optional resources to pass to the backend
- `metadata`: Configuration options including:
  - `vllm_api_key` or `vllmApiKey`: API key for secure VLLM deployments
  - `vllm_model` or `vllmModel`: Default model to use
  - `vllm_api_url` or `vllmApiUrl`: Base URL for the API (for self-hosted deployments)
  - `timeout`: Request timeout in milliseconds

### Methods

#### `createEndpointHandler(endpointUrl?: string): (data: VLLMRequestData) => Promise<VLLMResponse>`

Creates a handler for a specific endpoint.

#### `createVLLMChatEndpointHandler(endpointUrl?: string): (data: VLLMRequestData) => Promise<VLLMResponse>`

Creates a handler for a chat endpoint.

#### `testEndpoint(endpointUrl?: string, model?: string): Promise<boolean>`

Tests if an endpoint is available and working.

#### `testVLLMChatEndpoint(endpointUrl?: string, model?: string): Promise<boolean>`

Tests if a chat endpoint is available and working.

#### `makePostRequestVLLM(endpointUrl: string, data: VLLMRequestData, options?: VLLMRequestOptions): Promise<VLLMResponse>`

Makes a POST request to the VLLM server.

#### `makeStreamRequestVLLM(endpointUrl: string, data: VLLMRequestData, options?: VLLMRequestOptions): Promise<AsyncGenerator<VLLMResponse>>`

Makes a streaming request to the VLLM server.

#### `getModelInfo(model?: string): Promise<VLLMModelInfo>`

Gets information about a model.

#### `getModelStatistics(model?: string): Promise<VLLMModelStatistics>`

Gets statistics about a model.

#### `listLoraAdapters(): Promise<VLLMLoraAdapter[]>`

Lists available LoRA adapters.

#### `loadLoraAdapter(adapterData: {adapter_name: string, adapter_path: string, base_model: string}): Promise<any>`

Loads a LoRA adapter.

#### `setQuantization(model?: string, config: VLLMQuantizationConfig): Promise<any>`

Sets quantization for a model.

#### `processBatch(endpointUrl?: string, batchData: string[], model?: string, options?: VLLMRequestOptions): Promise<string[]>`

Processes a batch of prompts.

#### `processBatchWithMetrics(endpointUrl?: string, batchData: string[], model?: string, options?: VLLMRequestOptions): Promise<[string[], Record<string, any>]>`

Processes a batch of prompts and returns processing metrics.

#### `streamGeneration(endpointUrl?: string, prompt: string, model?: string, options?: VLLMRequestOptions): AsyncGenerator<string>`

Streams generation for a prompt.

#### `makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any>`

Makes a POST request to the VLLM server (BaseApiBackend implementation).

#### `makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Makes a streaming request to the VLLM server (BaseApiBackend implementation).

#### `chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse>`

Generates a chat completion from a list of messages.

#### `streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Generates a streaming chat completion.

#### `isCompatibleModel(model: string): boolean`

Checks if a model is compatible with VLLM.

## Interface Reference

### `VLLMRequestData`

Request data for VLLM API.

```typescript
interface VLLMRequestData {
  prompt?: string;
  messages?: Array<{ role: string; content: string | any[] }>;
  model?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  use_beam_search?: boolean;
  stop?: string | string[];
  n?: number;
  best_of?: number;
  stream?: boolean;
  [key: string]: any;
}
```

### `VLLMResponse`

Response from VLLM API.

```typescript
interface VLLMResponse {
  text?: string;
  texts?: string[];
  message?: {
    content: string;
    role: string;
    [key: string]: any;
  };
  choices?: Array<{
    text?: string;
    message?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    delta?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    index?: number;
    finish_reason?: string;
    [key: string]: any;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    [key: string]: any;
  };
  metadata?: {
    finish_reason?: string;
    model?: string;
    usage?: {
      prompt_tokens?: number;
      completion_tokens?: number;
      total_tokens?: number;
      [key: string]: any;
    };
    is_streaming?: boolean;
    [key: string]: any;
  };
  [key: string]: any;
}
```

### `VLLMModelInfo`

Model information.

```typescript
interface VLLMModelInfo {
  model: string;
  max_model_len: number;
  num_gpu: number;
  dtype: string;
  gpu_memory_utilization: number;
  quantization?: {
    enabled: boolean;
    method?: string;
    [key: string]: any;
  };
  lora_adapters?: string[];
  [key: string]: any;
}
```

### `VLLMModelStatistics`

Model statistics.

```typescript
interface VLLMModelStatistics {
  model: string;
  statistics: {
    requests_processed?: number;
    tokens_generated?: number;
    avg_tokens_per_request?: number;
    max_tokens_per_request?: number;
    avg_generation_time?: number;
    throughput?: number;
    errors?: number;
    uptime?: number;
    [key: string]: any;
  };
  [key: string]: any;
}
```

### `VLLMLoraAdapter`

LoRA adapter information.

```typescript
interface VLLMLoraAdapter {
  id: string;
  name: string;
  base_model: string;
  size_mb: number;
  active: boolean;
  [key: string]: any;
}
```

### `VLLMQuantizationConfig`

Quantization configuration.

```typescript
interface VLLMQuantizationConfig {
  enabled: boolean;
  method?: string;
  bits?: number;
  [key: string]: any;
}
```

### `VLLMRequestOptions`

Request options.

```typescript
interface VLLMRequestOptions extends ApiRequestOptions {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stop?: string | string[];
  [key: string]: any;
}
```

## Common Base Backend Interface

As a class that extends the `BaseApiBackend`, the VLLM backend implements all the standardized methods from the base class, ensuring compatibility with the unified API ecosystem.