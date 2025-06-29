# OpenAI Mini API Backend

The OpenAI Mini API backend is a lightweight, optimized version of the OpenAI client. It provides the core functionality of the OpenAI API with improved performance and reduced overhead.

## Overview

- Provides all essential OpenAI API features with a streamlined implementation
- Enhanced performance with circuit breaker pattern and request queuing
- Customizable configuration for timeout, retries, and debugging
- Support for all major OpenAI models and endpoints

## Installation

The OpenAI Mini backend is included in the IPFS Accelerate JavaScript SDK. No additional installation is required.

## Configuration

### API Key

To use the OpenAI Mini API, you need an API key. You can provide it in any of the following ways:

1. Pass it directly in the metadata when creating the backend instance:
   ```typescript
   import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';
   
   const client = new OpenAiMini({}, {
     openai_mini_api_key: 'your-api-key'
   });
   ```

2. Set it as an environment variable:
   ```bash
   # Node.js environment
   export OPENAI_API_KEY=your-api-key
   ```

### Custom Options

The OpenAI Mini client supports several configuration options:

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';

const client = new OpenAiMini({
  // API endpoint URL (defaults to standard OpenAI API)
  apiUrl: 'https://api.openai.com/v1',
  
  // Maximum number of retries for failed requests
  maxRetries: 3,
  
  // Request timeout in milliseconds
  requestTimeout: 30000,
  
  // Enable request queuing for rate limiting
  useRequestQueue: true,
  
  // Enable debug mode for additional logging
  debug: false
});
```

## Basic Usage

### Chat Completions

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const client = new OpenAiMini({}, {
  openai_mini_api_key: 'your-api-key'
});

// Define messages
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Hello! How are you today?' }
];

// Generate a chat completion
const response = await client.chat(messages, {
  model: 'gpt-3.5-turbo',
  temperature: 0.7,
  max_tokens: 500
});

console.log(response.content);
```

### Streaming Chat Completions

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const client = new OpenAiMini({}, {
  openai_mini_api_key: 'your-api-key'
});

// Define messages
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Write a short story about a robot learning to feel emotions.' }
];

// Generate a streaming chat completion
const stream = client.streamChat(messages, {
  model: 'gpt-3.5-turbo',
  temperature: 0.7
});

// Process the stream
for await (const chunk of stream) {
  process.stdout.write(chunk.content || '');
}
```

## Advanced Features

### File Upload

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';
import * as fs from 'fs';

// Create an instance with your API key
const client = new OpenAiMini({}, {
  openai_mini_api_key: 'your-api-key'
});

// Upload a file for fine-tuning
const fileResult = await client.uploadFile('/path/to/training_data.jsonl', {
  purpose: 'fine-tune',
  fileName: 'custom_name.jsonl'
});

console.log(`File uploaded with ID: ${fileResult.id}`);
```

### Text-to-Speech

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';
import * as fs from 'fs';

// Create an instance with your API key
const client = new OpenAiMini({}, {
  openai_mini_api_key: 'your-api-key'
});

// Convert text to speech
const audioBuffer = await client.textToSpeech(
  'Hello, this is a test of the text to speech API.',
  {
    model: 'tts-1',
    voice: 'alloy',
    speed: 1.0,
    response_format: 'mp3'
  }
);

// Save the audio to a file
fs.writeFileSync('output.mp3', audioBuffer);
```

### Speech-to-Text (Whisper)

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const client = new OpenAiMini({}, {
  openai_mini_api_key: 'your-api-key'
});

// Transcribe an audio file
const transcription = await client.transcribeAudio('/path/to/audio.mp3', {
  model: 'whisper-1',
  language: 'en',
  prompt: 'This is a clear recording of someone speaking.',
  response_format: 'text',
  temperature: 0.2
});

console.log(transcription);
```

### Image Generation (DALL-E)

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const client = new OpenAiMini({}, {
  openai_mini_api_key: 'your-api-key'
});

// Generate an image from a prompt
const imageResult = await client.generateImage(
  'A futuristic city with flying cars and tall skyscrapers at sunset',
  {
    model: 'dall-e-3',
    size: '1024x1024',
    quality: 'standard',
    style: 'vivid',
    n: 1
  }
);

// Get the image URL
console.log(`Generated image URL: ${imageResult.data[0].url}`);
```

## Error Handling

The OpenAI Mini backend throws standardized error objects with clear error messages:

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';

const client = new OpenAiMini({}, {
  openai_mini_api_key: 'your-api-key'
});

try {
  const response = await client.chat([{ role: 'user', content: 'Hello' }]);
  console.log(response.content);
} catch (error) {
  console.error(`Error: ${error.message}`);
  
  // Handle specific error types
  if (error.message.includes('API key')) {
    console.error('Authentication error - check your API key');
  } else if (error.message.includes('rate limit')) {
    console.error('Rate limit exceeded - slow down requests');
  }
}
```

## Compatibility

The OpenAI Mini backend is compatible with the following OpenAI models:

### Chat Models
- gpt-4o-mini (newest efficient model)
- gpt-4o (full capabilities model with vision)
- gpt-3.5-turbo, gpt-4, gpt-4-turbo

### Embedding Models
- text-embedding-3-small (1536 dimensions)
- text-embedding-3-large (3072 dimensions, higher quality)
- text-embedding-ada-002 (legacy)

### Image Models
- dall-e-3, dall-e-2

### Audio Models
- tts-1 (standard quality text-to-speech)
- tts-1-hd (high definition text-to-speech)
- whisper-1 (audio transcription and translation)

## Performance Optimizations

### Circuit Breaker Pattern

The backend implements a circuit breaker pattern to prevent cascading failures when the OpenAI API is experiencing issues:

```typescript
// Circuit breaker will automatically:
// 1. Track recent request failures
// 2. Open the circuit (stop sending requests) if too many failures occur
// 3. Gradually test the API with a half-open state before fully closing the circuit
```

### Request Queuing

The backend can queue requests when too many are sent simultaneously:

```typescript
import { OpenAiMini } from 'ipfs_accelerate_js/api_backends';

// Create an instance with custom queue settings
const client = new OpenAiMini({
  useRequestQueue: true
});

// Requests will be queued and processed according to their priority
```

## Comparison with Full OpenAI Backend

| Feature | OpenAI Mini | Full OpenAI Backend |
|---------|-------------|---------------------|
| Core API Features | ✅ | ✅ |
| Chat Completions | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Embeddings | ✅ via text-embedding-ada-002 | ✅ More models |
| File Upload | ✅ | ✅ |
| TTS/STT | ✅ | ✅ |
| Image Generation | ✅ | ✅ |
| Memory Usage | 🟢 Lower | 🟠 Higher |
| Startup Time | 🟢 Faster | 🟠 Slower |
| Advanced Features | 🟠 Basic | 🟢 Comprehensive |
| Debug Information | 🟠 Basic | 🟢 Detailed |

Choose OpenAI Mini when you want a lightweight, optimized implementation that covers all essential functionality while using fewer resources. Use the full OpenAI backend when you need the most comprehensive feature set and detailed debugging capabilities.