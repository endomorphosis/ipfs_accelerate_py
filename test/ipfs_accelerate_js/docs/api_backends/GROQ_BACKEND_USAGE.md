# Groq API Backend - Usage Guide

This document provides instructions on how to use the Groq API backend in the IPFS Accelerate JS framework.

## Introduction

The Groq API backend provides access to Groq's LLM models through their API. Groq offers high-performance language models with low latency, making them suitable for various applications. This backend supports all Groq API features including chat completions, with both streaming and non-streaming options.

## Prerequisites

To use the Groq API backend, you'll need:

1. A Groq API key (get one from [Groq Console](https://console.groq.com/))
2. Node.js v16 or higher
3. IPFS Accelerate JS framework

## Installation

The Groq API backend is included in the IPFS Accelerate JS package, so no additional installation is required.

```bash
npm install @ipfs-accelerate/js
```

## Basic Usage

### Initialize the Backend

```typescript
import { GroqBackend } from '@ipfs-accelerate/js/api_backends';

// Initialize with API key
const groqBackend = new GroqBackend({}, {
  groq_api_key: 'your-api-key-here',
  // Optional configuration
  groq_api_version: 'v1',
  max_retries: 3,
  initial_retry_delay: 1000,
  backoff_factor: 2,
  timeout: 30000
});

// Or use environment variable GROQ_API_KEY
// export GROQ_API_KEY=your-api-key-here
const groqBackendWithEnvKey = new GroqBackend();
```

### Chat Completions

```typescript
// Example chat completion
async function getChatCompletion() {
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What are the benefits of using LLMs in healthcare?' }
  ];
  
  try {
    const response = await groqBackend.chat(messages, {
      model: 'llama3-70b-8192', // Optional, defaults to "llama3-8b-8192"
      temperature: 0.7,         // Optional
      max_tokens: 1000,         // Optional
      top_p: 1,                 // Optional
    });
    
    console.log('Response:', response.content);
    console.log('Token usage:', response.usage);
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Streaming Chat Completions

```typescript
// Example streaming chat completion
async function getStreamingChatCompletion() {
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Tell me about quantum computing.' }
  ];
  
  try {
    const stream = groqBackend.streamChat(messages, {
      model: 'llama3-70b-8192', // Optional, defaults to "llama3-8b-8192"
      temperature: 0.7,         // Optional
      max_tokens: 1000,         // Optional
      top_p: 1,                 // Optional
    });
    
    let fullResponse = '';
    
    for await (const chunk of stream) {
      process.stdout.write(chunk.content || '');
      fullResponse += chunk.content || '';
    }
    
    console.log('\nFull response:', fullResponse);
  } catch (error) {
    console.error('Error:', error);
  }
}
```

## Advanced Usage

### Testing Endpoint Availability

```typescript
async function testConnection() {
  const isAvailable = await groqBackend.testEndpoint();
  console.log('Groq API is available:', isAvailable);
}
```

### Getting Available Models

```typescript
function listModels() {
  const models = groqBackend.getAvailableModels();
  console.log('Available models:', models);
  
  // Get info about a specific model
  const modelInfo = groqBackend.getModelInfo('llama3-70b-8192');
  console.log('Model info:', modelInfo);
}
```

### Getting Usage Statistics

```typescript
function checkUsage() {
  const stats = groqBackend.getUsageStats();
  console.log('API usage statistics:', stats);
  
  // Reset usage statistics if needed
  groqBackend.resetUsageStats();
}
```

### Creating Multiple Endpoints

```typescript
function createEndpoints() {
  // Create multiple endpoints with different configurations
  const endpointId1 = groqBackend.createEndpoint({
    model: 'llama3-8b-8192',
    maxConcurrentRequests: 2
  });
  
  const endpointId2 = groqBackend.createEndpoint({
    model: 'llama3-70b-8192',
    maxConcurrentRequests: 1
  });
  
  // Make request using a specific endpoint
  groqBackend.makeRequestWithEndpoint(endpointId1, {
    messages: [{ role: 'user', content: 'Hello' }]
  });
  
  // Get statistics for an endpoint
  const stats = groqBackend.getStats(endpointId1);
  console.log('Endpoint stats:', stats);
}
```

## Error Handling

The backend implements proper error handling with retries and backoff:

```typescript
try {
  const response = await groqBackend.chat([
    { role: 'user', content: 'Hello' }
  ]);
  console.log(response);
} catch (error) {
  if (error.isAuthError) {
    console.error('Authentication error - check your API key');
  } else if (error.isRateLimitError) {
    console.error('Rate limit exceeded, try again later');
  } else if (error.isTimeout) {
    console.error('Request timed out');
  } else if (error.isTransientError) {
    console.error('Temporary server error, try again later');
  } else {
    console.error('Unknown error:', error.message);
  }
}
```

## Best Practices

1. **API Key Security**: Never hardcode your API key in your application code. Use environment variables or a secure configuration management system.

2. **Model Selection**: Choose the appropriate model for your use case. The smaller models like `llama3-8b-8192` are faster and more cost-effective for simple tasks, while larger models like `llama3-70b-8192` provide better results for complex tasks.

3. **Error Handling**: Always implement proper error handling, as shown above.

4. **Rate Limits**: Be aware of Groq's rate limits. The backend automatically handles rate limiting with exponential backoff.

5. **Token Usage Monitoring**: Monitor your token usage using the `getUsageStats()` method to avoid unexpected costs.

## Supported Models

The backend supports all Groq API models, including:

- `llama3-8b-8192`: Meta's LLaMA 3 8B model
- `llama3-70b-8192`: Meta's LLaMA 3 70B model
- `mixtral-8x7b-32768`: Mixtral 8x7B with 32K context window
- `gemma2-9b-it`: Google's Gemma 2 9B instruction-tuned model
- `llama-3.3-70b-versatile`: Meta's LLaMA 3.3 70B versatile model with long context
- `qwen-2.5-32b`: Qwen 2.5 32B model with long context

And various other models as they become available through the Groq API.

## Additional Resources

- [Groq API Documentation](https://console.groq.com/docs/quickstart)
- [IPFS Accelerate JS Documentation](https://github.com/your-org/ipfs-accelerate-js)