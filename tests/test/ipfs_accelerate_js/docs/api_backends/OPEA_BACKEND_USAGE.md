# OpenAI Proxy API Extension (OPEA) Backend

The `OPEA` backend provides integration with OpenAI-compatible API proxies and self-hosted services that implement the OpenAI API interface. This includes various LLM servers, proxies, and middleware that conform to the OpenAI API specification.

This implementation extends the `BaseApiBackend` class, providing a standardized interface that works with the IPFS Accelerate API ecosystem. It can be used interchangeably with other API backends like OpenAI, Groq, Claude, etc., through the unified API interface.

## Features

- ChatGPT-style completion with OpenAI-compatible APIs
- Streaming text generation
- Chat interface with message formatting
- API key management for secure deployments
- Request tracking and statistics
- Exponential backoff and retries
- Circuit breaker pattern
- Queue management
- Support for various OpenAI API compatible backends

## Installation

The OPEA backend is included in the IPFS Accelerate JavaScript SDK. Ensure you have the `ipfs_accelerate_js` package installed.

## Basic Usage

```typescript
import { OPEA } from 'ipfs_accelerate_js/api_backends';

// Initialize with API key and URL
const opea = new OPEA({}, {
  opea_api_url: 'http://localhost:8000',
  opea_api_key: 'your_api_key',
  opea_model: 'gpt-3.5-turbo'
});

// Simple chat completion
async function chatWithModel() {
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain quantum entanglement in simple terms.' }
  ];
  
  const response = await opea.chat(messages, {
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
  
  for await (const chunk of opea.streamChat(messages, { temperature: 0.8 })) {
    process.stdout.write(chunk.content);
    if (chunk.done) break;
  }
}
```

## Using with the API Backend Factory

The OPEA backend can be used with the API backend factory for easy creation and model compatibility detection:

```typescript
import { createApiBackend, findCompatibleBackend } from 'ipfs_accelerate_js/api_backends';

// Create by name
const opea = createApiBackend('opea', {}, {
  opea_api_url: 'http://localhost:8000',
  opea_api_key: 'your_api_key',
  opea_model: 'gpt-3.5-turbo'
});

// Find compatible backend for a model
const modelName = 'gpt-3.5-turbo';
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
// For self-hosted OpenAI-compatible deployments
const customOPEA = new OPEA({}, {
  opea_api_key: 'your_api_key',
  opea_model: 'gpt-3.5-turbo',
  opea_api_url: 'http://localhost:8000' // Base URL for your OpenAI-compatible API
});
```

### Endpoint Handlers

```typescript
// Create endpoint handlers for custom endpoints
const opea = new OPEA();

// Create chat completions endpoint handler
const chatHandler = opea.createEndpointHandler(
  'http://localhost:8000/v1/chat/completions'
);

// Use the handler directly
const response = await chatHandler({
  messages: [{ role: 'user', content: 'Hello!' }],
  model: 'gpt-3.5-turbo',
  temperature: 0.7
});
```

### Streaming Support

```typescript
// Make a streaming request
const opea = new OPEA();

// Chat message to stream
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Write a short story about AI.' }
];

// Make the request and process the stream
const stream = await opea.makeStreamRequestOPEA(
  'http://localhost:8000/v1/chat/completions',
  {
    messages,
    model: 'gpt-3.5-turbo',
    temperature: 0.7,
    max_tokens: 500
  }
);

// Process the stream
for await (const chunk of stream) {
  if (chunk.choices && chunk.choices.length > 0) {
    const delta = chunk.choices[0].delta;
    if (delta && delta.content) {
      process.stdout.write(delta.content);
    }
  }
}
```

### Testing Endpoints

```typescript
// Test if an endpoint is available
const opea = new OPEA();
const isAvailable = await opea.testEndpoint();
console.log(`Endpoint available: ${isAvailable}`);

// Test with a specific URL
const isCustomAvailable = await opea.testEndpoint('http://localhost:8000/v1/chat/completions');
console.log(`Custom endpoint available: ${isCustomAvailable}`);
```

## API Reference

### Constructor

```typescript
new OPEA(resources?: ApiResources, metadata?: ApiMetadata)
```

**Parameters:**
- `resources`: Optional resources to pass to the backend
- `metadata`: Configuration options including:
  - `opea_api_key` or `opeaApiKey`: API key for OPEA
  - `opea_model` or `opeaModel`: Default model to use
  - `opea_api_url` or `opeaApiUrl`: Base URL for the API
  - `timeout`: Request timeout in milliseconds

### Methods

#### `createEndpointHandler(endpointUrl?: string): (data: OPEARequestData) => Promise<OPEAResponse>`

Creates a handler for a specific endpoint.

#### `testEndpoint(endpointUrl?: string): Promise<boolean>`

Tests if an endpoint is available and working.

#### `makePostRequestOPEA(endpointUrl: string, data: OPEARequestData, options?: OPEARequestOptions): Promise<OPEAResponse>`

Makes a POST request to the OPEA server.

#### `makeStreamRequestOPEA(endpointUrl: string, data: OPEARequestData, options?: OPEARequestOptions): Promise<AsyncGenerator<OPEAStreamChunk>>`

Makes a streaming request to the OPEA server.

#### `makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any>`

Makes a POST request to the OPEA server (BaseApiBackend implementation).

#### `makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Makes a streaming request to the OPEA server (BaseApiBackend implementation).

#### `chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse>`

Generates a chat completion from a list of messages.

#### `streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Generates a streaming chat completion.

#### `isCompatibleModel(model: string): boolean`

Checks if a model is compatible with OPEA.

## Interface Reference

### `OPEARequestData`

Request data for OPEA API.

```typescript
interface OPEARequestData {
  model?: string;
  messages?: Array<{ role: string; content: string | any[]; [key: string]: any }>;
  prompt?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string | string[];
  stream?: boolean;
  [key: string]: any;
}
```

### `OPEAResponse`

Response from OPEA API.

```typescript
interface OPEAResponse {
  choices?: Array<{
    message?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    text?: string;
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
  id?: string;
  created?: number;
  object?: string;
  model?: string;
  [key: string]: any;
}
```

### `OPEAStreamChunk`

Streaming chunk from OPEA API.

```typescript
interface OPEAStreamChunk {
  choices?: Array<{
    delta?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    index?: number;
    finish_reason?: string | null;
    [key: string]: any;
  }>;
  id?: string;
  created?: number;
  model?: string;
  object?: string;
  [key: string]: any;
}
```

### `OPEARequestOptions`

Request options.

```typescript
interface OPEARequestOptions extends ApiRequestOptions {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  [key: string]: any;
}
```

## Common Base Backend Interface

As a class that extends the `BaseApiBackend`, the OPEA backend implements all the standardized methods from the base class, ensuring compatibility with the unified API ecosystem.