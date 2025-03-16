# Gemini API Backend - Usage Guide

This document provides instructions on how to use the Gemini API backend in the IPFS Accelerate JS framework.

## Introduction

The Gemini API backend provides access to Google's Gemini language models through their API. Gemini models offer high-quality text generation, reasoning, and multimodal capabilities. This backend supports all Gemini API features including chat completions, with both streaming and non-streaming options.

## Prerequisites

To use the Gemini API backend, you'll need:

1. A Google AI Studio API key (get one from [Google AI Studio](https://aistudio.google.com/))
2. Node.js v16 or higher
3. IPFS Accelerate JS framework

## Installation

The Gemini API backend is included in the IPFS Accelerate JS package, so no additional installation is required.

```bash
npm install @ipfs-accelerate/js
```

## Basic Usage

### Initialize the Backend

```typescript
import { Gemini } from '@ipfs-accelerate/js/api_backends';

// Initialize with API key
const geminiBackend = new Gemini({}, {
  gemini_api_key: 'your-api-key-here',
  // Optional configuration
  max_retries: 3,
  initial_retry_delay: 1000,
  backoff_factor: 2,
  timeout: 30000
});

// Or use environment variable GEMINI_API_KEY
// export GEMINI_API_KEY=your-api-key-here
const geminiBackendWithEnvKey = new Gemini();
```

### Chat Completions

```typescript
// Example chat completion
async function getChatCompletion() {
  const messages = [
    { role: 'user', content: 'What are the key features of quantum computing?' }
  ];
  
  try {
    const response = await geminiBackend.chat(messages, {
      model: 'gemini-pro', // Optional, defaults to "gemini-pro"
      temperature: 0.7,    // Optional
      maxTokens: 1000,     // Optional
      topP: 0.9            // Optional
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
    { role: 'user', content: 'Explain neural networks in simple terms.' }
  ];
  
  try {
    const stream = geminiBackend.streamChat(messages, {
      model: 'gemini-pro', // Optional, defaults to "gemini-pro"
      temperature: 0.7,    // Optional
      maxTokens: 1000,     // Optional
      topP: 0.9            // Optional
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
  const isAvailable = await geminiBackend.testEndpoint();
  console.log('Gemini API is available:', isAvailable);
}
```

### Multi-turn Conversations

```typescript
async function multiTurnConversation() {
  const messages = [
    { role: 'user', content: 'What is machine learning?' },
    { role: 'assistant', content: 'Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from and make decisions based on data...' },
    { role: 'user', content: 'Can you give me some real-world applications?' }
  ];
  
  const response = await geminiBackend.chat(messages, {
    model: 'gemini-pro',
    maxTokens: 500
  });
  
  console.log('Response:', response.content);
}
```

### Creating Multiple Endpoints

```typescript
function createEndpoints() {
  // Create multiple endpoints with different configurations
  const endpointId1 = geminiBackend.createEndpoint({
    model: 'gemini-pro',
    maxConcurrentRequests: 2
  });
  
  const endpointId2 = geminiBackend.createEndpoint({
    model: 'gemini-pro-vision', // Vision model for multimodal capabilities
    maxConcurrentRequests: 1
  });
  
  // Make request using a specific endpoint
  geminiBackend.makeRequestWithEndpoint(endpointId1, {
    messages: [{ role: 'user', content: 'Hello' }]
  });
  
  // Get statistics for an endpoint
  const stats = geminiBackend.getStats(endpointId1);
  console.log('Endpoint stats:', stats);
}
```

## Error Handling

The backend implements proper error handling with retries and backoff:

```typescript
try {
  const response = await geminiBackend.chat([
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

2. **Model Selection**: Choose the appropriate model for your use case:
   - `gemini-pro`: Text generation, reasoning, and more
   - `gemini-pro-vision`: Multimodal capabilities (text and images)
   - `gemini-1.5-pro`: Newer version with improved capabilities
   - `gemini-1.5-flash`: Fast, efficient model for simple tasks

3. **Token Usage**: Be mindful of token limits. Different models have different context lengths and pricing.

4. **Error Handling**: Always implement proper error handling, as shown above.

5. **Rate Limits**: Be aware of Google's rate limits. The backend automatically handles rate limiting with exponential backoff.

6. **Token Usage Monitoring**: Monitor your token usage using the usage reporting to avoid unexpected costs.

## Supported Models

The backend supports all Gemini API models, including:

- `gemini-pro`: Google's Gemini Pro language model
- `gemini-pro-vision`: Multimodal model that can understand both text and images
- `gemini-1.5-pro`: Newer version with improved capabilities
- `gemini-1.5-flash`: Faster, more efficient version for simple tasks
- `palm-2`: Legacy model (may be deprecated)

And various other models as they become available through the Google AI Studio API.

## Request Format Conversion

The Gemini API uses a different format than the standard OpenAI-style format. The backend automatically handles the conversion between formats, so you can use the same interface as other API backends.

### Standard Format (What You Use)
```typescript
const messages = [
  { role: 'user', content: 'Hello' },
  { role: 'assistant', content: 'Hi there!' },
  { role: 'user', content: 'How are you?' }
];
```

### Gemini Format (Handled Automatically)
```typescript
const geminiFormat = {
  contents: [
    { role: 'user', parts: [{ text: 'Hello' }] },
    { role: 'model', parts: [{ text: 'Hi there!' }] },
    { role: 'user', parts: [{ text: 'How are you?' }] }
  ]
};
```

## Additional Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs/gemini_api_overview)
- [Google AI Studio](https://aistudio.google.com/)
- [IPFS Accelerate JS Documentation](https://github.com/your-org/ipfs-accelerate-js)