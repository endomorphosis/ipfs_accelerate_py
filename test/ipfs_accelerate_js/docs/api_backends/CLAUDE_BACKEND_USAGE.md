# Claude API Backend - Usage Guide

This document provides instructions on how to use the Claude (Anthropic) API backend in the IPFS Accelerate JS framework.

## Introduction

The Claude API backend provides access to Anthropic's Claude language models through their API. Claude models excel at a wide range of natural language processing tasks, including text generation, summarization, and creative writing. This backend supports all Claude API features including chat completions, with both streaming and non-streaming options.

The backend implements the Messages API format using the latest API version and includes support for streaming responses, system messages, and all Claude model variants.

## Prerequisites

To use the Claude API backend, you'll need:

1. An Anthropic API key (get one from [Anthropic Console](https://console.anthropic.com/))
2. Node.js v16 or higher
3. IPFS Accelerate JS framework

## Installation

The Claude API backend is included in the IPFS Accelerate JS package, so no additional installation is required.

```bash
npm install ipfs-accelerate-js
```

## Basic Usage

### Initialize the Backend

```typescript
import { Claude } from 'ipfs-accelerate-js/api_backends';

// Initialize with API key
const claudeBackend = new Claude({}, {
  claude_api_key: 'your-api-key-here',
  // Optional configuration
  max_retries: 3,
  initial_retry_delay: 1000,
  backoff_factor: 2,
  timeout: 30000
});

// Or use environment variable ANTHROPIC_API_KEY
// export ANTHROPIC_API_KEY=your-api-key-here
const claudeBackendWithEnvKey = new Claude();
```

### Chat Completions

```typescript
// Example chat completion
async function getChatCompletion() {
  const messages = [
    { role: 'user', content: 'What are the benefits of quantum computing?' }
  ];
  
  try {
    const response = await claudeBackend.chat(messages, {
      model: 'claude-3-haiku-20240307', // Optional, defaults to "claude-3-haiku-20240307"
      temperature: 0.7,                // Optional
      max_tokens: 1000,                // Optional
      system: 'You are a helpful assistant specialized in quantum computing.' // Optional system message
    });
    
    // Response content is an array of content blocks
    // Each content block has a type (usually 'text') and content
    console.log('Response:', response.content);
    
    // For simple text extraction:
    const textContent = response.content
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('');
    console.log('Text content:', textContent);
    
    // Token usage information
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
    const stream = claudeBackend.streamChat(messages, {
      model: 'claude-3-haiku-20240307', // Optional, defaults to "claude-3-haiku-20240307"
      temperature: 0.7,                // Optional
      max_tokens: 1000,                // Optional
      system: 'You are a helpful assistant specialized in explaining complex concepts simply.' // Optional
    });
    
    let fullResponse = '';
    
    for await (const chunk of stream) {
      // Different chunk types have different properties
      if (chunk.type === 'delta') {
        process.stdout.write(chunk.content || '');
        fullResponse += chunk.content || '';
      } else if (chunk.type === 'start') {
        console.log('Assistant response started');
      } else if (chunk.type === 'stop') {
        console.log('\nAssistant response completed');
      }
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
  const isAvailable = await claudeBackend.testEndpoint();
  console.log('Claude API is available:', isAvailable);
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
  
  const response = await claudeBackend.chat(messages, {
    model: 'claude-3-haiku-20240307',
    max_tokens: 500
  });
  
  // Extract text content from the response
  const textContent = response.content
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('');
  
  console.log('Response:', textContent);
}
```

### Using System Messages

Claude supports system messages to guide the assistant's behavior:

```typescript
async function chatWithSystemMessage() {
  const messages = [
    { role: 'user', content: 'Write a short poem about artificial intelligence.' }
  ];
  
  const response = await claudeBackend.chat(messages, {
    model: 'claude-3-haiku-20240307',
    system: 'You are a brilliant poet who specializes in creating concise and thoughtful poetry.',
    temperature: 0.9
  });
  
  // Extract text content from the response
  const textContent = response.content
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('');
  
  console.log('Response:', textContent);
}
```

### Working with Content Blocks

Claude's responses include structured content blocks:

```typescript
async function handleContentBlocks() {
  const messages = [
    { role: 'user', content: 'Write a brief summary of quantum computing.' }
  ];
  
  const response = await claudeBackend.chat(messages);
  
  // Process content blocks by type
  for (const block of response.content) {
    if (block.type === 'text') {
      console.log('Text:', block.text);
    }
    // Future support for other block types like 'image', etc.
  }
}
```

### Creating Multiple Endpoints

```typescript
function createEndpoints() {
  // Create multiple endpoints with different configurations
  const endpointId1 = claudeBackend.createEndpoint({
    model: 'claude-3-haiku-20240307',
    maxConcurrentRequests: 2
  });
  
  const endpointId2 = claudeBackend.createEndpoint({
    model: 'claude-3-opus-20240229',
    maxConcurrentRequests: 1
  });
  
  // Make request using a specific endpoint
  claudeBackend.makeRequestWithEndpoint(endpointId1, {
    messages: [{ role: 'user', content: 'Hello' }]
  });
  
  // Get statistics for an endpoint
  const stats = claudeBackend.getStats(endpointId1);
  console.log('Endpoint stats:', stats);
}
```

## Error Handling

The backend implements robust error handling with retries and exponential backoff:

```typescript
try {
  const response = await claudeBackend.chat([
    { role: 'user', content: 'Hello' }
  ]);
  
  // Extract text content from the response
  const textContent = response.content
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('');
  
  console.log(textContent);
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

### Automatic Retries

The Claude backend implements automatic retries for transient errors:

```typescript
// Configure retry behavior in the constructor
const claudeBackend = new Claude({}, {
  claude_api_key: process.env.ANTHROPIC_API_KEY,
  max_retries: 5,                // Maximum number of retry attempts
  initial_retry_delay: 1000,     // Initial delay in milliseconds
  backoff_factor: 2              // Exponential backoff multiplier
});
```

## Best Practices

1. **API Key Security**: Never hardcode your API key in your application code. Use environment variables or a secure configuration management system.

2. **Model Selection**: Choose the appropriate model for your use case:
   - `claude-3-haiku-20240307`: Fastest, most cost-effective model for simpler tasks
   - `claude-3-sonnet-20240229`: Balance of speed, cost, and intelligence
   - `claude-3-opus-20240229`: Most capable model for complex tasks

3. **System Messages**: Use system messages to guide Claude's behavior and set the tone for responses.

4. **Error Handling**: Always implement proper error handling to gracefully handle API issues.

5. **Rate Limits**: Be aware of Anthropic's rate limits. The backend automatically handles rate limiting with exponential backoff.

6. **Token Usage Monitoring**: Monitor your token usage using the response usage data to avoid unexpected costs.

7. **Content Block Handling**: Always use proper content block type checking as Claude API may introduce new block types over time.

8. **Streaming for Long Responses**: Use streaming for long responses to improve user experience with progressive rendering.

9. **Circuit Breaker Pattern**: For production applications, the backend implements circuit breaker patterns to prevent cascading failures.

10. **Timeouts**: Set appropriate timeouts based on your application's needs and the expected response time of the models.

## Supported Models

The backend supports all Claude API models, including:

- `claude-3-haiku-20240307`: Anthropic's fastest Claude 3 model, ideal for high-volume, simple tasks
- `claude-3-sonnet-20240229`: Mid-tier Claude 3 model balancing speed and capabilities
- `claude-3-opus-20240229`: Most capable Claude 3 model for complex tasks
- `claude-3-5-sonnet-20240620`: Latest Claude 3.5 model with improved capabilities
- `claude-2.1`: Previous generation model (deprecated)
- `claude-instant-1.2`: Legacy fast model (deprecated)

And various other models as they become available through the Anthropic API.

## Implementation Details

The Claude backend uses the latest Anthropic API:
- API Endpoint: `https://api.anthropic.com/v1/messages`
- API Version: `2023-06-01`
- Messages Beta: `messages-2023-12-15`

This ensures compatibility with all Claude features, including content blocks and streaming.

## Additional Resources

- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Claude 3.5 Documentation](https://docs.anthropic.com/claude/docs/claude-3-5-models)
- [Claude Messages API](https://docs.anthropic.com/claude/reference/messages_post)
- [IPFS Accelerate JS Documentation](https://github.com/ipfs/ipfs-accelerate-js)