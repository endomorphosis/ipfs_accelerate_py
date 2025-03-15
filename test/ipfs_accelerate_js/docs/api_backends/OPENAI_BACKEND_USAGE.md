# OpenAI API Backend

The OpenAI API backend provides a TypeScript implementation for interacting with the OpenAI API. This backend supports all major OpenAI API features including:

- Chat completions
- Function calling (tools)
- Streaming responses
- Embeddings
- Content moderation
- Image generation (DALL-E)
- Text-to-speech and speech-to-text

## Installation

The OpenAI backend is included in the IPFS Accelerate JavaScript SDK. No additional installation is required.

## Configuration

### API Key

To use the OpenAI API, you need an API key. You can provide it in any of the following ways:

1. Pass it directly in the metadata when creating the backend instance:
   ```typescript
   import { OpenAI } from 'ipfs_accelerate_js/api_backends';
   
   const openai = new OpenAI({}, {
     openai_api_key: 'your-api-key'
   });
   ```

2. Set it as an environment variable:
   ```bash
   # Node.js environment
   export OPENAI_API_KEY=your-api-key
   ```

## Basic Usage

### Chat Completions

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Define messages
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Hello! How are you today?' }
];

// Generate a chat completion
const response = await openai.chat(messages, {
  model: 'gpt-4o',
  temperature: 0.7,
  maxTokens: 500
});

console.log(response.content);
```

### Streaming Chat Completions

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Define messages
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Write a short story about a robot learning to feel emotions.' }
];

// Generate a streaming chat completion
const stream = openai.streamChat(messages, {
  model: 'gpt-4o',
  temperature: 0.7
});

// Process the stream
for await (const chunk of stream) {
  process.stdout.write(chunk.content || '');
}
```

### Function Calling (Tools)

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Define tool specifications
const tools = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather in a location',
      parameters: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
            description: 'The city and state, e.g. San Francisco, CA'
          },
          unit: {
            type: 'string',
            enum: ['celsius', 'fahrenheit'],
            description: 'The temperature unit to use'
          }
        },
        required: ['location']
      }
    }
  }
];

// Define messages
const messages = [
  { role: 'user', content: 'What is the weather like in San Francisco?' }
];

// Generate a chat completion with function calling
const response = await openai.chat(messages, {
  model: 'gpt-4o',
  tools,
  toolChoice: 'auto'
});

// Check if function was called
if (response.tool_calls) {
  for (const toolCall of response.tool_calls) {
    if (toolCall.function.name === 'get_weather') {
      const args = JSON.parse(toolCall.function.arguments);
      console.log(`Getting weather for ${args.location}`);
      
      // Call your actual weather function here
      const weatherData = await getWeather(args.location, args.unit);
      
      // Add the function result to the conversation
      messages.push({
        role: 'assistant',
        content: null,
        tool_calls: [toolCall]
      });
      
      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        name: 'get_weather',
        content: JSON.stringify(weatherData)
      });
      
      // Get the final response
      const finalResponse = await openai.chat(messages, {
        model: 'gpt-4o'
      });
      
      console.log(finalResponse.content);
    }
  }
} else {
  console.log(response.content);
}
```

### Embeddings

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Generate embeddings for a text
const embeddings = await openai.embedding('This is a sample text to embed', {
  model: 'text-embedding-3-small'
});

console.log(`Embedding dimensions: ${embeddings[0].length}`);
```

### Content Moderation

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Check if text violates content policy
const moderationResult = await openai.moderation('Some text to check for policy violations');

console.log(`Flagged: ${moderationResult.results[0].flagged}`);
if (moderationResult.results[0].flagged) {
  console.log('Categories:');
  for (const [category, flagged] of Object.entries(moderationResult.results[0].categories)) {
    if (flagged) {
      console.log(`- ${category}: ${moderationResult.results[0].category_scores[category].toFixed(3)}`);
    }
  }
}
```

### Image Generation (DALL-E)

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Generate an image from a prompt
const imageResult = await openai.textToImage(
  'A futuristic city with flying cars and tall skyscrapers',
  {
    model: 'dall-e-3',
    size: '1024x1024',
    style: 'vivid',
    quality: 'hd'
  }
);

// Get the image URL
const imageUrl = imageResult.data[0].url;
console.log(`Generated image URL: ${imageUrl}`);
```

### Text-to-Speech

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import * as fs from 'fs';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Convert text to speech
const audioBuffer = await openai.textToSpeech(
  'Hello, this is a test of the text to speech API.',
  'alloy',
  {
    model: 'tts-1',
    responseFormat: 'mp3',
    speed: 1.0
  }
);

// Save the audio to a file
fs.writeFileSync('output.mp3', Buffer.from(audioBuffer));
console.log('Audio saved to output.mp3');
```

### Speech-to-Text (Whisper)

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import * as fs from 'fs';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Read audio file
const audioFile = fs.readFileSync('input.mp3');
const audioBlob = new Blob([audioFile], { type: 'audio/mp3' });

// Convert speech to text
const transcription = await openai.speechToText(audioBlob, {
  model: 'whisper-1',
  language: 'en'
});

console.log(transcription.text);
```

## Advanced Usage

### Handling Rate Limits and Retries

The OpenAI backend includes built-in support for handling rate limits with exponential backoff and automatic retries. You can customize the retry behavior:

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with custom retry settings
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key',
  maxRetries: 5,
  initialRetryDelay: 1000, // ms
  backoffFactor: 2
});
```

### Circuit Breaker

The backend implements a circuit breaker pattern to prevent cascading failures when the OpenAI API is experiencing issues:

```typescript
// The circuit breaker will automatically:
// 1. Track recent request failures
// 2. Open the circuit (stop sending requests) if too many failures occur
// 3. Gradually test the API with a half-open state before fully closing the circuit
```

### Request Queuing

The backend can queue requests when too many are sent simultaneously:

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with custom queue settings
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key',
  maxConcurrentRequests: 10,
  queueSize: 100
});
```

## Compatibility

The OpenAI backend is compatible with the following OpenAI models:

- Chat models: gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo
- Embedding models: text-embedding-3-small, text-embedding-3-large
- Image models: dall-e-3
- Audio models: tts-1, whisper-1
- Moderation models: text-moderation-latest

## Error Handling

The backend throws standardized error objects with helpful properties:

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

try {
  const response = await openai.chat([{ role: 'user', content: 'Hello' }]);
  console.log(response.content);
} catch (error) {
  console.error(`Error: ${error.message}`);
  
  // Check error type
  if (error.isAuthError) {
    console.error('Authentication error - check your API key');
  } else if (error.isRateLimitError) {
    console.error(`Rate limit exceeded. Retry after ${error.retryAfter} seconds`);
  } else if (error.isTransientError) {
    console.error('Temporary server error, retry later');
  }
  
  console.error(`Status code: ${error.statusCode}`);
  console.error(`Error type: ${error.type}`);
}
```