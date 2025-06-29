# OpenAI API Backend

The OpenAI API backend provides a TypeScript implementation for interacting with the OpenAI API. This backend supports all major OpenAI API features including:

- Chat completions with latest models (GPT-4o, GPT-4o-mini)
- Function calling with parallel execution
- Streaming responses
- Embeddings (text-embedding-3-small/large)
- Content moderation
- Image generation (DALL-E)
- Enhanced text-to-speech with multiple voice options
- Speech-to-text with word-level timestamps
- Voice agents for conversational interactions
- Comprehensive metrics collection for distributed testing

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

### Chat Completions with GPT-4o and GPT-4o-mini

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

// Generate a chat completion with GPT-4o
const response = await openai.chat(messages, {
  model: 'gpt-4o',
  temperature: 0.7,
  maxTokens: 500
});

console.log('GPT-4o response:', response.content);

// Generate a more cost-effective response with GPT-4o-mini
const economicResponse = await openai.chat(messages, {
  model: 'gpt-4o-mini',
  temperature: 0.7,
  maxTokens: 500
});

console.log('GPT-4o-mini response:', economicResponse.content);

// Get model information
console.log('GPT-4o max tokens:', openai.getModelMaxTokens('gpt-4o'));
console.log('GPT-4o-mini max tokens:', openai.getModelMaxTokens('gpt-4o-mini'));
console.log('GPT-4o supports vision:', openai.supportsVision('gpt-4o'));
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

### Parallel Function Calling (Tools)

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Define messages
const messages = [
  { role: 'user', content: 'What\'s the weather and news headline for New York City and San Francisco?' }
];

// Define functions to be executed in parallel
const functions = {
  get_weather: async (args: { location: string }) => {
    console.log(`Getting weather for ${args.location}...`);
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 500));
    // Return mock weather data
    return {
      location: args.location,
      temperature: Math.round(70 + Math.random() * 20),
      conditions: ['sunny', 'partly cloudy', 'cloudy', 'rainy'][Math.floor(Math.random() * 4)]
    };
  },
  get_news: async (args: { location: string, category?: string }) => {
    console.log(`Getting news for ${args.location} (category: ${args.category || 'general'})...`);
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 800));
    // Return mock news data
    return {
      location: args.location,
      headline: `Major development announced in ${args.location}`,
      source: 'News API',
      category: args.category || 'general'
    };
  }
};

// Execute entire conversation with function calls in a single method
const response = await openai.chatWithFunctions(messages, functions, {
  model: 'gpt-4o',
  temperature: 0.7,
  maxRounds: 3,           // Maximum rounds of function calling
  functionTimeout: 5000,  // Timeout for each function call
  failOnFunctionError: false  // Continue even if functions fail
});

console.log('Final response:', response.content);
```

### Traditional Function Calling (Tools)

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

### Embeddings with Latest Models

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Generate embeddings with text-embedding-3-small
const smallEmbeddings = await openai.embedding('This is a sample text to embed', {
  model: 'text-embedding-3-small'
});

console.log(`Small embedding dimensions: ${smallEmbeddings[0].length}`);

// Generate high-quality embeddings with text-embedding-3-large
const largeEmbeddings = await openai.embedding('This is a sample text to embed', {
  model: 'text-embedding-3-large'
});

console.log(`Large embedding dimensions: ${largeEmbeddings[0].length}`);
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

### Enhanced Text-to-Speech with Voice Options

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import { OpenAIVoiceType, OpenAIAudioFormat } from 'ipfs_accelerate_js/api_backends/openai/types';
import * as fs from 'fs';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Convert text to speech with multiple voice options
const voices = [
  OpenAIVoiceType.ALLOY,   // Neutral voice
  OpenAIVoiceType.ECHO,    // Male voice
  OpenAIVoiceType.FABLE,   // Male storytelling voice
  OpenAIVoiceType.ONYX,    // Male authoritative voice
  OpenAIVoiceType.NOVA,    // Female voice
  OpenAIVoiceType.SHIMMER  // Female soft voice
];

// Sample message to convert to speech
const message = "Hello, this is a demonstration of the different voice options available.";

// Generate speech with each voice type
for (const voice of voices) {
  const audioBuffer = await openai.textToSpeech(
    message,
    voice,
    {
      model: 'tts-1-hd',
      responseFormat: OpenAIAudioFormat.MP3,
      speed: 1.0
    }
  );
  
  // Save the audio to a file
  fs.writeFileSync(`output_${voice}.mp3`, Buffer.from(audioBuffer));
  console.log(`Audio with voice ${voice} saved to output_${voice}.mp3`);
}
```

### Speech-to-Text with Word-Level Timestamps

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import { OpenAITranscriptionFormat } from 'ipfs_accelerate_js/api_backends/openai/types';
import * as fs from 'fs';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Read audio file
const audioFile = fs.readFileSync('input.mp3');
const audioBlob = new Blob([audioFile], { type: 'audio/mp3' });

// Convert speech to text with word-level timestamps
const transcription = await openai.speechToText(audioBlob, {
  model: 'whisper-1',
  language: 'en',
  responseFormat: OpenAITranscriptionFormat.VERBOSE_JSON,
  timestamp_granularities: ['word']
});

console.log('Transcription text:', transcription.text);

// Display word-level timestamps
if (transcription.words && transcription.words.length > 0) {
  console.log('\nWord timestamps:');
  transcription.words.forEach(word => {
    console.log(`"${word.word}": ${word.start}s to ${word.end}s (probability: ${word.probability.toFixed(2)})`);
  });
}
```

### Audio Translation

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import * as fs from 'fs';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Read non-English audio file
const audioFile = fs.readFileSync('non_english_input.mp3');
const audioBlob = new Blob([audioFile], { type: 'audio/mp3' });

// Translate audio to English
const translation = await openai.translateAudio(audioBlob, {
  model: 'whisper-1'
});

console.log('Translation:', translation.text);
```

### Voice Agent for Conversational Interactions

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import { OpenAIVoiceType, OpenAIAudioFormat } from 'ipfs_accelerate_js/api_backends/openai/types';
import * as fs from 'fs';

// Create an instance with your API key
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Create a voice agent
const voiceAgent = openai.createVoiceAgent(
  "You are a helpful assistant with expertise in technology. Be concise and informative.",
  {
    voice: OpenAIVoiceType.NOVA,
    model: 'tts-1-hd',
    speed: 1.1,
    format: OpenAIAudioFormat.MP3
  },
  {
    chatModel: 'gpt-4o-mini',
    temperature: 0.7
  }
);

// Process a text query and get spoken response
const result = await voiceAgent.processText(
  "What are the key differences between TypeScript and JavaScript?"
);

// Save the audio response
fs.writeFileSync('voice_response.mp3', Buffer.from(result.audioResponse));
console.log('Text response:', result.textResponse);
console.log('Audio response saved to voice_response.mp3');

// Process audio input (if you have an audio file)
const audioFile = fs.readFileSync('question.mp3');
const audioBlob = new Blob([audioFile], { type: 'audio/mp3' });

const audioResult = await voiceAgent.processAudio(audioBlob);
fs.writeFileSync('voice_response_2.mp3', Buffer.from(audioResult.audioResponse));
console.log('Response to audio question:', audioResult.textResponse);

// Get conversation history
console.log('Conversation messages:', voiceAgent.getMessages());

// Reset the conversation
voiceAgent.reset("You are a technical support agent specializing in web development.");
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

### Circuit Breaker Pattern

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

### Metrics Collection for Distributed Testing

The OpenAI backend now includes comprehensive metrics collection for integration with the distributed testing framework:

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Create an instance
const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Make several API calls
await openai.chat([{ role: 'user', content: 'Hello' }], { model: 'gpt-4o-mini' });
await openai.embedding('Test embedding', { model: 'text-embedding-3-small' });

// Metrics are automatically collected and can be accessed
const metrics = (openai as any).recentRequests;
console.log(`Number of tracked requests: ${Object.keys(metrics).length}`);

// The metrics include:
// - Request type (chat, embedding, tts, stt, etc.)
// - Start timestamp
// - Duration
// - Success status
// - Model used
// - Token counts (input, output, total)
// - Error type (if applicable)
// - Retry count
// - Request size
// - Response size

// Metrics are automatically reported to the distributed testing framework
// if it's available in the environment
```

## Compatibility

The OpenAI backend is compatible with the following OpenAI models:

### Chat Models
- GPT-4o (latest model with vision capabilities)
- GPT-4o-mini (efficient, economical alternative to GPT-4o)
- GPT-4-turbo, GPT-4
- GPT-3.5-turbo

### Embedding Models
- text-embedding-3-small (1536 dimensions)
- text-embedding-3-large (3072 dimensions, higher quality)
- text-embedding-ada-002 (legacy)

### Image Models
- dall-e-3

### Audio Models
- tts-1 (standard quality text-to-speech)
- tts-1-hd (high definition text-to-speech)
- whisper-1 (audio transcription and translation)

### Moderation Models
- text-moderation-latest

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

## Integration with Distributed Testing Framework

The enhanced OpenAI client integrates automatically with the IPFS Accelerate Distributed Testing Framework to provide real-time performance metrics:

```typescript
// Metrics are automatically collected for all API calls
// No additional code needed for basic integration

// For advanced configuration:
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

const openai = new OpenAI({}, {
  openai_api_key: 'your-api-key',
  // Enable detailed metrics
  requestTracking: true,
  // Configure metrics cleanup
  metricsRetentionMs: 3600000, // 1 hour
  // Custom metrics reporting
  metricsPrefix: 'custom_openai_'
});

// The metrics will be accessible in the Distributed Testing Dashboard
// showing performance trends, error rates, and resource utilization
```