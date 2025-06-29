# OllamaClean Backend Usage Guide

## Introduction

The OllamaClean backend provides an OpenAI-compatible interface for Ollama models. It simplifies working with locally deployed Ollama models by providing an API that follows the OpenAI format, making it easy to switch between OpenAI's API and locally hosted Ollama models without changing your code.

> **Note**: The OllamaClean backend is currently under development. This documentation describes the intended functionality and usage patterns.

## Prerequisites

- An API key for the OllamaClean service (when available)
- Node.js 16 or later
- The IPFS Accelerate JS SDK installed in your project

## Installation

The OllamaClean backend is included in the IPFS Accelerate JS SDK. Install the package:

```bash
npm install ipfs-accelerate-js
```

## Basic Usage

### Initializing the Backend

```typescript
import { OllamaClean } from 'ipfs-accelerate-js/api_backends/ollama_clean';

// Initialize with API key
const ollamaClean = new OllamaClean(
  {}, // Resources (not needed for basic usage)
  {
    // API key (will be required when the service is available)
    ollama_clean_api_key: process.env.OLLAMA_CLEAN_API_KEY || 'your_api_key',
    // Default model (optional)
    model: 'llama3'
  }
);
```

### Chat Completion

```typescript
// Define your messages
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is machine learning?' }
];

// Get a completion
const response = await ollamaClean.chat(messages, {
  temperature: 0.7,
  max_tokens: 150
});

console.log(response.text);
```

### Streaming Chat Completion

For real-time responses, you can use streaming:

```typescript
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Write a short story about a robot.' }
];

// Get a streaming completion
const stream = ollamaClean.streamChat(messages, {
  temperature: 0.8,
  max_tokens: 200
});

// Process the stream
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

## Advanced Usage

### Specifying Different Models

You can override the default model for specific requests:

```typescript
const response = await ollamaClean.chat(
  [{ role: 'user', content: 'Explain quantum computing briefly.' }],
  {
    model: 'mistral',
    temperature: 0.5,
    max_tokens: 100
  }
);
```

### Using Direct API Format

For advanced users who need more control:

```typescript
import { OllamaCleanRequest } from 'ipfs-accelerate-js/api_backends/ollama_clean/types';

// Create a request in the direct API format
const directRequest: OllamaCleanRequest = {
  model: 'llama3',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' }
  ],
  temperature: 0.5,
  max_tokens: 50
};

// Use the createEndpointHandler method (when implemented)
const handler = ollamaClean.createEndpointHandler();
const response = await handler(directRequest);
```

## Error Handling

The OllamaClean backend will implement comprehensive error handling for various scenarios:

```typescript
try {
  const response = await ollamaClean.chat(messages);
  console.log(response.text);
} catch (error) {
  if (error.name === 'AuthenticationError') {
    console.error('API key is invalid');
  } else if (error.name === 'RateLimitError') {
    console.error('Rate limit exceeded');
  } else if (error.name === 'ModelNotFoundError') {
    console.error('The specified model was not found');
  } else {
    console.error('An unexpected error occurred:', error);
  }
}
```

## API Reference

### Constructor Options

```typescript
new OllamaClean(resources: Record<string, any>, metadata: ApiMetadata)
```

- `resources`: Additional resources needed by the backend (optional)
- `metadata`: Configuration options
  - `ollama_clean_api_key`: Your API key for the OllamaClean service
  - `model`: Default model to use (optional)

### Methods

#### `chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse>`

Generates a completion for the given messages.

- `messages`: Array of message objects with `role` and `content`
- `options`: Additional options
  - `model`: Override the default model
  - `temperature`: Control randomness (0.0 to 1.0)
  - `max_tokens`: Maximum number of tokens to generate
  - `top_p`: Control diversity via nucleus sampling
  - Other options as supported by the API

#### `streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Generates a streaming completion for the given messages.

- `messages`: Array of message objects with `role` and `content`
- `options`: Same as for `chat()`

#### `testEndpoint(): Promise<boolean>`

Tests the connection to the API endpoint.

#### `createEndpointHandler(): Function`

Creates a function that handles direct API requests.

## Compatibility with OpenAI

A key feature of the OllamaClean backend is its compatibility with the OpenAI API format. This allows you to use the same code with both OpenAI and Ollama models. The differences are handled internally by the backend.

Example of switching between backends:

```typescript
// Using OpenAI
const openai = new OpenAI({}, { openai_api_key: 'your_openai_key' });
const openaiResponse = await openai.chat(messages);

// Using OllamaClean with the same message format
const ollamaClean = new OllamaClean({}, { ollama_clean_api_key: 'your_key' });
const ollamaResponse = await ollamaClean.chat(messages);
```

## Best Practices

1. **Environment Variables**: Store your API key in environment variables rather than hardcoding them.

```typescript
const ollamaClean = new OllamaClean({}, {
  ollama_clean_api_key: process.env.OLLAMA_CLEAN_API_KEY
});
```

2. **Error Handling**: Always implement proper error handling to manage API issues gracefully.

3. **Resource Management**: For applications that make many requests, implement proper resource management and rate limiting awareness.

4. **Model Selection**: Choose the appropriate model for your task to balance performance and cost.

## Planned Features

The OllamaClean backend will include these planned features:

- Full compatibility with OpenAI API formats
- Support for all Ollama models
- Automatic handling of token counting
- Configurable model caching
- Comprehensive error handling
- Request rate limiting and queue management
- Circuit breaker pattern for fault tolerance

## Additional Resources

- [OllamaClean API Documentation](https://docs.ollamaclean.com) (planned)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## Support

For issues, feature requests, or questions about the OllamaClean backend, please open an issue in the IPFS Accelerate JS repository.