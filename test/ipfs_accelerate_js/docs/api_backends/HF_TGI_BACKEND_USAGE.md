# HuggingFace Text Generation Inference (TGI) Backend

The `HfTgi` backend provides integration with HuggingFace's Text Generation Inference API, supporting both the hosted HuggingFace Inference API and self-hosted TGI deployments.

This implementation extends the `BaseApiBackend` class, providing a standardized interface that works with the IPFS Accelerate API ecosystem. It can be used interchangeably with other API backends like OpenAI, Groq, Claude, etc., through the unified API interface.

## Features

- Text generation with HuggingFace models
- Streaming text generation
- Chat interface with message formatting
- API key management
- Multiple endpoint management
- Request tracking and statistics
- Exponential backoff and retries
- Circuit breaker pattern
- Support for batch processing
- Queue management

## Installation

The HF TGI backend is included in the IPFS Accelerate JavaScript SDK. Ensure you have the `ipfs_accelerate_js` package installed.

## Basic Usage

```typescript
import { HfTgi } from 'ipfs_accelerate_js/api_backends';

// Initialize with API key and model
const hfTgi = new HfTgi({}, {
  hf_api_key: 'your_huggingface_api_key', 
  model_id: 'mistralai/Mistral-7B-Instruct-v0.2'
});

// Simple text generation
async function generateText() {
  const response = await hfTgi.generateText(
    'mistralai/Mistral-7B-Instruct-v0.2',
    'Tell me about quantum computing',
    { max_new_tokens: 100, temperature: 0.7 }
  );
  
  console.log(response.generated_text);
}

// Chat interface
async function chatWithModel() {
  const messages = [
    { role: 'system', content: 'You are a helpful assistant specializing in quantum physics.' },
    { role: 'user', content: 'Explain quantum entanglement in simple terms.' }
  ];
  
  const response = await hfTgi.chat(messages, {
    temperature: 0.7,
    max_new_tokens: 200
  });
  
  console.log(response.text);
}

// Streaming chat
async function streamingChat() {
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Write a short poem about spring.' }
  ];
  
  for await (const chunk of hfTgi.streamChat(messages, { temperature: 0.8 })) {
    process.stdout.write(chunk.content);
    if (chunk.done) break;
  }
}
```

## Using with the API Backend Factory

The HF TGI backend can be used with the API backend factory for easy creation and model compatibility detection:

```typescript
import { createApiBackend, findCompatibleBackend } from 'ipfs_accelerate_js/api_backends';

// Create by name
const hfTgi = createApiBackend('hf_tgi', {}, {
  hf_api_key: 'your_huggingface_api_key',
  model_id: 'mistralai/Mistral-7B-Instruct-v0.2'
});

// Find compatible backend for a model
const modelName = 'mistralai/Mistral-7B-Instruct-v0.2';
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
// For self-hosted TGI deployments
const customHfTgi = new HfTgi({}, {
  hf_api_key: 'your_api_key', // May not be required for local deployments
  model_id: 'my-model',
  api_base: 'http://localhost:8080' // Base URL for your TGI deployment
});
```

### Multiple Endpoints

```typescript
// Create multiple endpoints with different settings
const hfTgi = new HfTgi();

// Create endpoints for different models
const endpoint1 = hfTgi.createEndpoint({
  id: 'mistral-endpoint',
  api_key: 'your_api_key',
  model_id: 'mistralai/Mistral-7B-Instruct-v0.2',
  max_concurrent_requests: 5
});

const endpoint2 = hfTgi.createEndpoint({
  id: 'llama-endpoint',
  api_key: 'your_api_key',
  model_id: 'meta-llama/Llama-2-7b-chat-hf',
  max_concurrent_requests: 3
});

// Use a specific endpoint
const response = await hfTgi.chat(messages, { endpointId: endpoint1 });

// Get statistics for an endpoint
const stats = hfTgi.getStats(endpoint1);
console.log(`Total requests: ${stats.total_requests}`);
```

### Custom Request Formatting

```typescript
// Create an endpoint handler
const endpointUrl = 'https://api-inference.huggingface.co/models/gpt2';
const handler = hfTgi.createRemoteTextGenerationEndpointHandler(endpointUrl, 'your_api_key');

// Format a request with specific parameters
const response = await hfTgi.formatRequest(
  handler,
  'Once upon a time',
  100,  // max_new_tokens
  0.7,  // temperature
  0.9,  // top_p
  40,   // top_k
  1.2   // repetition_penalty
);

console.log(response.generated_text);
```

### Testing Endpoints

```typescript
// Test if an endpoint is available
const isAvailable = await hfTgi.testEndpoint();
console.log(`Endpoint available: ${isAvailable}`);

// Test a specific endpoint
const isTgiAvailable = await hfTgi.testTgiEndpoint(
  'https://api-inference.huggingface.co/models/gpt2',
  'your_api_key',
  'gpt2'
);
console.log(`TGI endpoint available: ${isTgiAvailable}`);
```

## API Reference

### Constructor

```typescript
new HfTgi(resources?: Record<string, any>, metadata?: ApiMetadata)
```

**Parameters:**
- `resources`: Optional resources to pass to the backend
- `metadata`: Configuration options including:
  - `hf_api_key` or `hfApiKey`: HuggingFace API key
  - `model_id`: Default model to use
  - `api_base`: Base URL for the API (for self-hosted deployments)

### Methods

#### `createEndpoint(params: Partial<HfTgiEndpoint>): string`

Creates a new endpoint with specific settings, returning the endpoint ID.

**Parameters:**
- `params`: Endpoint configuration including:
  - `id`: Optional ID for the endpoint
  - `api_key`: API key for this endpoint
  - `model_id`: Model ID for this endpoint
  - Various queue and retry settings

#### `getEndpoint(endpointId?: string): HfTgiEndpoint`

Gets an endpoint by ID or creates a default one if not found.

#### `updateEndpoint(endpointId: string, params: Partial<HfTgiEndpoint>): HfTgiEndpoint`

Updates settings for an existing endpoint.

#### `getStats(endpointId?: string): HfTgiStats | any`

Gets usage statistics for an endpoint or global stats.

#### `resetStats(endpointId?: string): void`

Resets usage statistics for an endpoint or globally.

#### `makeRequestWithEndpoint(endpointId: string, data: any, requestId?: string): Promise<any>`

Makes a request using a specific endpoint.

#### `generateText(modelId: string, inputs: string, parameters?: any, apiKey?: string, requestId?: string, endpointId?: string): Promise<any>`

Generates text with the specified model.

#### `streamGenerate(modelId: string, inputs: string, parameters?: any, apiKey?: string, requestId?: string, endpointId?: string): AsyncGenerator<any>`

Generates a streaming text response.

#### `chat(messages: Message[], options?: HfTgiChatOptions): Promise<ChatCompletionResponse>`

Generates a chat completion from a list of messages.

#### `streamChat(messages: Message[], options?: HfTgiChatOptions): AsyncGenerator<StreamChunk>`

Generates a streaming chat completion.

#### `formatRequest(handler: (data: any) => Promise<any>, text: string, max_new_tokens?: number, temperature?: number, top_p?: number, top_k?: number, repetition_penalty?: number): Promise<any>`

Formats a request with specific parameters for a handler.

#### `createEndpointHandler(endpointUrl?: string, apiKey?: string): (data: any) => Promise<any>`

Creates a handler for a specific endpoint.

#### `createRemoteTextGenerationEndpointHandler(endpointUrl?: string, apiKey?: string): (data: any) => Promise<any>`

Creates a handler for a remote text generation endpoint.

#### `testEndpoint(endpointUrl?: string, apiKey?: string, modelName?: string): Promise<boolean>`

Tests if an endpoint is available and working.

#### `testTgiEndpoint(endpointUrl: string, apiKey: string, modelName: string): Promise<any>`

Tests a specific TGI endpoint.

## Interface Reference

### `HfTgiResponse`

Response from the HuggingFace TGI API.

```typescript
interface HfTgiResponse {
  generated_text?: string;
  token?: {
    text: string;
    id?: number;
    logprob?: number;
    special?: boolean;
  };
  details?: {
    finish_reason: string;
    logprobs?: Array<{
      token: string;
      logprob: number;
    }>;
    generated_tokens: number;
  };
}
```

### `HfTgiRequest`

Request to the HuggingFace TGI API.

```typescript
interface HfTgiRequest {
  inputs: string;
  parameters?: {
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    return_full_text?: boolean;
    do_sample?: boolean;
    repetition_penalty?: number;
    seed?: number;
    watermark?: boolean;
    [key: string]: any;
  };
  stream?: boolean;
  options?: {
    use_cache?: boolean;
    wait_for_model?: boolean;
    [key: string]: any;
  };
}
```

### `HfTgiChatOptions`

Options for chat generation.

```typescript
interface HfTgiChatOptions {
  model?: string;
  temperature?: number;
  max_new_tokens?: number;
  top_p?: number;
  top_k?: number;
  return_full_text?: boolean;
  do_sample?: boolean;
  repetition_penalty?: number;
  endpointId?: string;
  requestId?: string;
  [key: string]: any;
}
```

### `HfTgiEndpoint`

Configuration for an endpoint. Extends the base ApiEndpoint interface.

```typescript
interface HfTgiEndpoint extends ApiEndpoint {
  // Standard properties from ApiEndpoint
  id: string;
  apiKey: string;
  model?: string;
  maxConcurrentRequests?: number;
  queueSize?: number;
  maxRetries?: number;
  initialRetryDelay?: number;
  backoffFactor?: number;
  timeout?: number;
  
  // HF TGI specific properties
  api_key: string;
  model_id: string;
  endpoint_url: string;
  max_retries?: number;
  max_concurrent_requests?: number;
  initial_retry_delay?: number;
  backoff_factor?: number;

  // Tracking properties  
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  current_requests: number;
  queue_processing: boolean;
  request_queue: Array<any>;
  last_request_at: number | null;
  created_at: number;
}
```

### `HfTgiStats`

Statistics for an endpoint.

```typescript
interface HfTgiStats {
  endpoint_id: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  created_at: number;
  last_request_at: number | null;
  current_queue_size: number;
  current_requests: number;
}
```

## Common Base Backend Interface

As a class that extends the `BaseApiBackend`, the HfTgi backend implements all the standardized methods from the base class, ensuring compatibility with the unified API ecosystem.