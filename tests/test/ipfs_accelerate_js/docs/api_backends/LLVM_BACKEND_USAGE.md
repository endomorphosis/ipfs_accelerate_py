# LLVM Backend Usage Guide

The LLVM backend provides a TypeScript interface for interacting with LLVM-based inference servers. This guide covers installation, configuration, and usage examples for integrating with LLVM APIs.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [API Reference](#api-reference)
  - [Initialization](#initialization)
  - [Model Operations](#model-operations)
  - [Inference](#inference)
  - [Chat](#chat)
- [Advanced Features](#advanced-features)
  - [Request Queue Management](#request-queue-management)
  - [Retry Mechanism](#retry-mechanism)
  - [Error Handling](#error-handling)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

The LLVM backend is included as part of the IPFS Accelerate JavaScript SDK. You can install it via npm:

```bash
npm install ipfs-accelerate
```

## Configuration

The LLVM backend can be configured using options passed to the constructor or through environment variables.

### Constructor Options

```typescript
import { LLVM } from 'ipfs-accelerate/api_backends/llvm';
import { LlvmOptions } from 'ipfs-accelerate/api_backends/llvm/types';

const options: LlvmOptions = {
  base_url: 'http://localhost:8000', // LLVM server URL
  max_concurrent_requests: 5,        // Maximum concurrent requests
  max_retries: 3,                    // Maximum retry attempts
  retry_delay: 1000,                 // Retry delay in milliseconds
  queue_size: 100                    // Maximum queue size
};

const llvm = new LLVM(options);
```

### Environment Variables

You can also configure the backend using environment variables:

- `LLVM_API_KEY`: API key for authentication
- `LLVM_BASE_URL`: Base URL of the LLVM server
- `LLVM_DEFAULT_MODEL`: Default model ID to use

Example:

```typescript
// Set environment variables
process.env.LLVM_API_KEY = 'your_api_key';
process.env.LLVM_BASE_URL = 'http://your-llvm-server:8000';
process.env.LLVM_DEFAULT_MODEL = 'llvm-model-name';

// Create backend with environment-based configuration
const llvm = new LLVM();
```

## Basic Usage

Here's a quick example of how to use the LLVM backend:

```typescript
import { LLVM } from 'ipfs-accelerate/api_backends/llvm';

async function main() {
  // Initialize the LLVM backend
  const llvm = new LLVM({
    base_url: 'http://localhost:8000'
  });
  
  // Set API key (if not provided via environment variables)
  llvm.setApiKey('your_api_key');
  
  // List available models
  const models = await llvm.listModels();
  console.log('Available models:', models.models);
  
  // Run inference
  const result = await llvm.runInference(
    'llvm-text-model',
    'What is the capital of France?',
    { max_tokens: 100 }
  );
  
  console.log('Inference result:', result.outputs);
}

main().catch(console.error);
```

## API Reference

### Initialization

```typescript
// Create a new LLVM backend
const llvm = new LLVM(options, metadata);
```

Parameters:
- `options`: Configuration options for the LLVM client
  - `base_url`: Base URL for the LLVM API server
  - `max_concurrent_requests`: Maximum concurrent requests (default: 10)
  - `max_retries`: Maximum retry attempts (default: 3)
  - `retry_delay`: Retry delay in milliseconds (default: 1000)
  - `queue_size`: Maximum queue size (default: 100)
- `metadata`: API metadata
  - `llvm_api_key`: API key for authentication
  - `llvm_default_model`: Default model ID to use

### Model Operations

#### List Models

Retrieves a list of available models from the LLVM server.

```typescript
const models = await llvm.listModels();
console.log(models.models); // Array of model IDs
```

#### Get Model Information

Retrieves detailed information about a specific model.

```typescript
const modelInfo = await llvm.getModelInfo('model-id');
console.log(modelInfo.status);       // Model status
console.log(modelInfo.details);      // Optional model details
```

### Inference

Runs inference with a specific model and input data.

```typescript
// Run inference with text input
const textResult = await llvm.runInference(
  'model-id',
  'What is the capital of France?',
  {
    max_tokens: 100,
    temperature: 0.7,
    top_p: 0.95
  }
);

// Run inference with structured input
const structuredResult = await llvm.runInference(
  'model-id',
  {
    text: 'What is the capital of France?',
    options: {
      format: 'json',
      include_references: true
    }
  },
  { max_tokens: 150 }
);
```

Parameters:
- `modelId`: ID of the model to use
- `inputs`: Input data (string or structured object)
- `options`: Inference options
  - `max_tokens`: Maximum tokens to generate
  - `temperature`: Temperature for controlling randomness
  - `top_k`: Sample top-k most likely tokens
  - `top_p`: Nucleus sampling parameter
  - `params`: Custom model parameters

### Chat

Provides a chat interface compatible with chat-based models.

```typescript
const chatResponse = await llvm.chat(
  'model-id',
  [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is machine learning?' }
  ],
  {
    max_tokens: 200,
    temperature: 0.8
  }
);

console.log(chatResponse.content); // Assistant's response
```

Parameters:
- `model`: ID of the model to use
- `messages`: Array of chat messages
  - Each message has `role` ('user', 'system', 'assistant') and `content`
- `options`: Inference options (same as runInference)

## Advanced Features

### Request Queue Management

The LLVM backend includes a sophisticated request queue management system with priority levels:

- `HIGH`: High-priority requests jump to the front of the queue
- `NORMAL`: Standard priority (default)
- `LOW`: Low-priority requests are processed last

This system:
1. Limits the number of concurrent requests to avoid overwhelming the server
2. Provides a queue for pending requests
3. Prioritizes requests based on their importance
4. Implements a circuit breaker pattern to handle server overload

The queue parameters can be configured:

```typescript
const llvm = new LLVM({
  max_concurrent_requests: 5,  // Maximum parallel requests
  queue_size: 50               // Maximum pending requests
});
```

### Retry Mechanism

The backend implements an exponential backoff retry mechanism:

```typescript
const llvm = new LLVM({
  max_retries: 3,     // Maximum retry attempts
  retry_delay: 1000   // Base delay in milliseconds
});
```

The retry delay increases exponentially with each attempt:
- 1st retry: 1000ms
- 2nd retry: 2000ms
- 3rd retry: 4000ms

### Error Handling

The backend implements comprehensive error handling. Errors are consistently formatted and include:

- Request-specific information (operation, model, etc.)
- Server response details when available
- HTTP status codes (for network errors)
- Retry information (attempts made, next retry time)

## Examples

See the full example demonstrating all features:

```typescript
// Import the required modules
import { LLVM } from 'ipfs-accelerate/api_backends/llvm';
import { ChatMessage } from 'ipfs-accelerate/api_backends/types';

async function main() {
  // Initialize the backend
  const llvm = new LLVM({
    base_url: 'http://localhost:8000',
    max_concurrent_requests: 5,
    max_retries: 3
  });
  
  // Set API key
  llvm.setApiKey('your_api_key');
  
  // Test connection
  const connected = await llvm.testEndpoint();
  console.log('Connected:', connected);
  
  // List models
  const models = await llvm.listModels();
  console.log('Available models:', models.models);
  
  // Get model info
  const modelInfo = await llvm.getModelInfo('llvm-model-1');
  console.log('Model status:', modelInfo.status);
  
  // Run inference
  const inferenceResult = await llvm.runInference(
    'llvm-model-1',
    'Translate to French: Hello, how are you?',
    { max_tokens: 100 }
  );
  console.log('Translation:', inferenceResult.outputs);
  
  // Chat interface
  const messages: ChatMessage[] = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is machine learning?' }
  ];
  
  const chatResponse = await llvm.chat('llvm-model-1', messages, {
    max_tokens: 200
  });
  
  console.log('Chat response:', chatResponse.content);
}

main().catch(console.error);
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   
   If you encounter connection errors, verify:
   - The LLVM server is running and accessible
   - The base_url is correct
   - Network connectivity between your application and the server

2. **Authentication Errors**
   
   If you see authentication errors:
   - Verify your API key is correct
   - Ensure the API key is being properly passed to the server

3. **Model Not Found**
   
   If you get "model not found" errors:
   - Use `listModels()` to verify available models
   - Check model naming format (should match LLVM server)
   - Verify the model is loaded on the server

### Debugging

Enable more detailed logging for troubleshooting:

```typescript
const llvm = new LLVM({
  ...options,
  debug: true  // Enable debug logging
});
```

You can also check the request count for monitoring purposes:

```typescript
// Access private property for debugging
// @ts-ignore
console.log('Total requests:', llvm.requestCount);
```