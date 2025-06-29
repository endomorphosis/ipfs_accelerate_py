# API Backend Interface Reference

This document provides a comprehensive reference for the `BaseApiBackend` interface used across all API backends in the IPFS Accelerate JavaScript SDK.

## Overview

The `BaseApiBackend` is an abstract base class that all API backends extend. It provides a standardized interface for interacting with various API services, handling common functionality such as:

- Authentication and API key management
- Request queueing and rate limiting
- Circuit breaker pattern for fault tolerance
- Request retries with exponential backoff
- Error handling and standardization
- Resource and metadata management
- Streaming support
- Request tracking and statistics

## Class Hierarchy

```
BaseApiBackend (abstract)
├── OpenAI
├── OpenAI Mini
├── Claude
├── Gemini
├── Groq
├── HF TGI
├── HF TEI
├── HF TGI Unified
├── HF TEI Unified
├── VLLM
├── VLLM Unified
├── Ollama
├── Ollama Clean
├── OVMS
├── S3 Kit
└── LLVM
```

## Constructor and Initialization

```typescript
constructor(resources: ApiResources = {}, metadata: ApiMetadata = {})
```

### Parameters

- `resources`: A record of resources needed by the backend
- `metadata`: Configuration options including API keys, model defaults, and behavior settings

### Example

```typescript
const backend = new SomeBackend(
  {}, // Resources (optional)
  {
    api_key: 'your-api-key',
    model: 'default-model',
    maxRetries: 5,
    timeout: 60000
  }
);
```

## Abstract Methods

These methods must be implemented by all backend classes:

### `protected abstract getApiKey(metadata: ApiMetadata): string`

Retrieves the API key from the provided metadata or environment variables.

### `protected abstract getDefaultModel(): string`

Returns the default model to use for this API backend.

### `abstract createEndpointHandler(): (data: any) => Promise<any>`

Creates a function that can handle direct API requests, allowing for low-level API access.

### `abstract testEndpoint(): Promise<boolean>`

Tests the connection to the API endpoint, returning true if successful.

### `abstract makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any>`

Makes a POST request to the API endpoint with the given data.

### `abstract makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Makes a streaming request to the API endpoint.

### `abstract chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse>`

Generates a chat completion for the given messages.

### `abstract streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Generates a streaming chat completion for the given messages.

## Common Methods

These methods are implemented in the base class and available to all backends:

### `createEndpoint(endpoint: Partial<ApiEndpoint>): string`

Creates a new API endpoint for multiplexing, returning an endpoint ID.

```typescript
const endpointId = backend.createEndpoint({
  apiKey: 'alternative-api-key',
  model: 'specific-model',
  maxConcurrentRequests: 10
});
```

### `getStats(endpointId: string): ApiEndpointStats`

Gets statistics for a specific endpoint.

```typescript
const stats = backend.getStats(endpointId);
console.log(`Requests: ${stats.requests}, Success: ${stats.success}, Errors: ${stats.errors}`);
```

### `makeRequestWithEndpoint(endpointId: string, data: any, options?: ApiRequestOptions): Promise<any>`

Makes a request using a specific endpoint.

```typescript
const response = await backend.makeRequestWithEndpoint(
  endpointId,
  { 
    model: 'gpt-4',
    messages: [{ role: 'user', content: 'Hello' }]
  }
);
```

### `isCompatibleModel(model: string): boolean`

Checks if a model is compatible with this API backend.

```typescript
if (backend.isCompatibleModel('gpt-4')) {
  // Use the model
}
```

## Protected Methods

These methods are available to derived classes:

### `protected trackRequestResult(success: boolean, requestId?: string, error?: Error): void`

Tracks request results for monitoring and circuit breaker functionality.

### `protected processQueue(): Promise<void>`

Processes the request queue, respecting concurrency limits and circuit breaker state.

### `protected createApiError(message: string, statusCode?: number, type?: string): ApiError`

Creates a standardized error object with appropriate properties.

### `protected retryableRequest<T>(requestFn: () => Promise<T>, maxRetries?: number, initialDelay?: number, backoffFactor?: number): Promise<T>`

Performs a request with automatic retries using exponential backoff.

## Key Properties

These properties control the behavior of the backend:

### Request Management

- `protected queueEnabled: boolean`: Controls whether request queueing is enabled
- `protected requestQueue: RequestQueueItem[]`: Queue of pending requests
- `protected currentRequests: number`: Number of current in-flight requests
- `protected maxConcurrentRequests: number`: Maximum number of concurrent requests
- `protected queueSize: number`: Maximum queue size

### Circuit Breaker

- `protected circuitState: 'OPEN' | 'CLOSED' | 'HALF-OPEN'`: Current state of the circuit breaker
  - `'CLOSED'`: Normal operation, requests are processed
  - `'OPEN'`: Circuit is broken, requests are rejected
  - `'HALF-OPEN'`: Testing if the service has recovered

### Retry Strategy

- `protected maxRetries: number`: Maximum number of retries
- `protected initialRetryDelay: number`: Initial delay before first retry (ms)
- `protected backoffFactor: number`: Multiplier for exponential backoff
- `protected timeout: number`: Request timeout (ms)

### Request Tracking

- `protected requestTracking: boolean`: Controls whether request tracking is enabled
- `protected recentRequests: Record<string, { timestamp: number; success: boolean; error?: string; }>`: Recent request history

## Interface Types

### ApiResources

```typescript
interface ApiResources {
  [key: string]: any;
}
```

A record of resources needed by the backend.

### ApiMetadata

```typescript
interface ApiMetadata {
  [key: string]: any;
}
```

Configuration options including API keys, model defaults, and behavior settings.

### ApiEndpoint

```typescript
interface ApiEndpoint {
  id: string;
  apiKey: string;
  model?: string;
  maxConcurrentRequests?: number;
  queueSize?: number;
  maxRetries?: number;
  initialRetryDelay?: number;
  backoffFactor?: number;
  timeout?: number;
  [key: string]: any;
}
```

Configuration for an API endpoint.

### ApiEndpointStats

```typescript
interface ApiEndpointStats {
  requests: number;
  success: number;
  errors: number;
  [key: string]: any;
}
```

Statistics for an API endpoint.

### Message

```typescript
interface Message {
  role: string;
  content: string | any[];
  [key: string]: any;
}
```

A message in a chat completion request.

### ChatCompletionResponse

```typescript
interface ChatCompletionResponse {
  id?: string;
  content?: any;
  role?: string;
  model?: string;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    [key: string]: any;
  };
  [key: string]: any;
}
```

The response from a chat completion request.

### StreamChunk

```typescript
interface StreamChunk {
  content?: string;
  role?: string;
  type?: string;
  delta?: any;
  [key: string]: any;
}
```

A chunk of data from a streaming response.

### ApiRequestOptions

```typescript
interface ApiRequestOptions {
  signal?: AbortSignal;
  requestId?: string;
  timeout?: number;
  [key: string]: any;
}
```

Options for an API request.

### ApiError

```typescript
interface ApiError extends Error {
  statusCode?: number;
  type?: string;
  isRateLimitError?: boolean;
  isAuthError?: boolean;
  isTransientError?: boolean;
  isTimeout?: boolean;
  retryAfter?: number;
  [key: string]: any;
}
```

A standardized error object.

### RequestQueueItem

```typescript
interface RequestQueueItem {
  data: any;
  apiKey: string;
  requestId: string;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  options?: ApiRequestOptions;
}
```

An item in the request queue.

## Circuit Breaker Pattern

The `BaseApiBackend` implements the circuit breaker pattern to prevent cascading failures and provide fault tolerance. This pattern has three states:

1. **CLOSED**: Normal operation. Requests are processed as usual.

2. **OPEN**: Circuit is broken. Requests are rejected to prevent overloading a failing service.

3. **HALF-OPEN**: Testing recovery. A limited number of requests are allowed through to see if the service has recovered.

The circuit breaker transitions between states based on error rates:

- When the error rate exceeds 50% in a one-minute window, the circuit opens
- After a delay, the circuit transitions to half-open to test if the service has recovered
- If a request succeeds in the half-open state, the circuit closes
- If a request fails in the half-open state, the circuit opens again

## Request Queue Management

The `BaseApiBackend` includes a request queue system that:

1. Limits concurrent requests to prevent overloading the API
2. Queues requests when concurrency limits are reached
3. Respects circuit breaker state
4. Processes requests in FIFO order
5. Handles request timeouts
6. Tracks request success/failure for monitoring

## Retry Strategy

The `BaseApiBackend` implements a retry strategy with exponential backoff:

1. Initial retry delay (configurable)
2. Exponential increase in delay between retries (configurable)
3. Maximum number of retries (configurable)
4. Respect for Retry-After headers when provided by the API
5. Jitter to prevent thundering herd problems
6. No retries for authentication errors

## Usage Example

```typescript
import { SomeBackend } from 'ipfs-accelerate-js/api_backends/some_backend';

async function example() {
  // Create a backend instance
  const backend = new SomeBackend(
    {}, // Resources
    {
      api_key: 'your-api-key',
      model: 'default-model',
      maxRetries: 3,
      timeout: 30000
    }
  );
  
  // Test the endpoint
  const isConnected = await backend.testEndpoint();
  if (!isConnected) {
    console.error('Could not connect to the API');
    return;
  }
  
  // Basic chat completion
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello, how are you?' }
  ];
  
  try {
    const response = await backend.chat(messages, {
      temperature: 0.7,
      max_tokens: 100
    });
    
    console.log(response.content);
  } catch (error) {
    console.error('Error:', error.message);
  }
  
  // Streaming chat completion
  const streamingMessages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Write a short poem.' }
  ];
  
  try {
    const stream = backend.streamChat(streamingMessages, {
      temperature: 0.8,
      max_tokens: 200
    });
    
    for await (const chunk of stream) {
      process.stdout.write(chunk.delta || '');
    }
  } catch (error) {
    console.error('Streaming error:', error.message);
  }
}
```

## Best Practices

1. **API Keys**: Store API keys in environment variables, not in code
2. **Error Handling**: Always wrap API calls in try/catch blocks
3. **Streaming**: Use streaming for real-time responses and better user experience
4. **Request Options**: Set appropriate timeout and retry values based on your use case
5. **Circuit Breaker**: Understand that errors may temporarily prevent requests from being processed
6. **Resource Management**: Close unused connections and streams
7. **Model Selection**: Use compatible models for each backend
8. **Rate Limiting**: Be aware of API rate limits and set concurrency appropriately

## Implementation Details

When implementing a new API backend, you must:

1. Extend the `BaseApiBackend` class
2. Implement all abstract methods
3. Override any base methods that need custom behavior
4. Create appropriate type definitions
5. Create tests, examples, and documentation

For detailed implementation guidance, see the [API Backend Development Guide](./API_BACKEND_DEVELOPMENT.md).