# OpenAI API Enhancements (July 2025)

This document describes the enhanced OpenAI API implementations for both Python and TypeScript in the IPFS Accelerate framework. These enhancements bring the latest OpenAI capabilities to both language implementations with consistent interfaces and advanced features.

## Key Enhancements

- **Latest Model Support**: Added support for GPT-4o, GPT-4o-mini and other latest models
- **Enhanced Voice Options**: Full support for all OpenAI voice types (alloy, echo, fable, onyx, nova, shimmer)
- **Advanced Speech-to-Text**: Word-level timestamps and improved language detection
- **Voice Agents**: Conversational agents with voice input/output capabilities
- **Parallel Function Calling**: Optimized execution of function calls in parallel
- **Comprehensive Metrics**: Detailed performance metrics for the distributed testing framework
- **Improved Error Handling**: Enhanced error recovery and circuit breaker pattern
- **Resource Optimization**: Memory efficient implementations for embedded environments

## Python Implementation

The Python implementation (`openai_api.py`) provides a comprehensive client for OpenAI's API with the following features:

### Core Features

```python
# Initialize the client
from ipfs_accelerate_py.api_backends.openai_api import openai_api

client = openai_api(api_key="your-api-key")

# Chat with latest models
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    model="gpt-4o-mini",  # Cost-effective alternative to gpt-4o
    temperature=0.7
)

# Embeddings with latest models
embeddings = client.embeddings(
    text="Sample text for embedding",
    model="text-embedding-3-large"  # Higher quality embeddings
)
```

### Voice Features

```python
# Enhanced Text-to-Speech with multiple voice options
from ipfs_accelerate_py.api_backends.openai_api import VoiceType, AudioFormat

# Available voice types
# VoiceType.ALLOY - Neutral voice
# VoiceType.ECHO - Male voice
# VoiceType.FABLE - Male storytelling voice
# VoiceType.ONYX - Male authoritative voice
# VoiceType.NOVA - Female voice
# VoiceType.SHIMMER - Female soft voice

audio_data = client.text_to_speech(
    text="Hello, this is a demonstration of text-to-speech capabilities.",
    voice=VoiceType.NOVA,
    model="tts-1-hd",
    response_format=AudioFormat.MP3,
    speed=1.1
)

# Save the audio data
with open("output.mp3", "wb") as f:
    f.write(audio_data)

# Speech-to-Text with word-level timestamps
from ipfs_accelerate_py.api_backends.openai_api import TranscriptionFormat

with open("input.mp3", "rb") as audio_file:
    transcription = client.speech_to_text(
        audio_file=audio_file,
        model="whisper-1",
        response_format=TranscriptionFormat.VERBOSE_JSON,
        timestamp_granularities=["word"]
    )

# Access word-level timestamps
if transcription.get("words"):
    for word in transcription["words"]:
        print(f"Word: {word['word']}, Start: {word['start']}s, End: {word['end']}s")

# Audio Translation
with open("non_english.mp3", "rb") as audio_file:
    translation = client.translate_audio(
        audio_file=audio_file,
        model="whisper-1"
    )
```

### Voice Agent

```python
# Create a voice agent for conversational interactions
voice_agent = client.create_voice_agent(
    initial_prompt="You are a helpful assistant with expertise in technology.",
    voice_settings={
        "voice": VoiceType.NOVA,
        "model": "tts-1-hd",
        "speed": 1.1,
        "format": AudioFormat.MP3
    },
    chat_settings={
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
)

# Process text input and get spoken response
text_result = voice_agent.process_text(
    "What are the key differences between Python and JavaScript?"
)

# Save the audio response
with open("response.mp3", "wb") as f:
    f.write(text_result.audio_response)

print("Text response:", text_result.text_response)

# Process audio input
with open("question.mp3", "rb") as audio_file:
    audio_result = voice_agent.process_audio(audio_file)

# Access conversation history
conversation = voice_agent.get_messages()

# Reset the conversation
voice_agent.reset("You are a technical support agent specializing in programming.")
```

### Parallel Function Calling

```python
# Define functions to be called in parallel
def get_weather(location, unit="celsius"):
    # In a real implementation, this would call a weather API
    import time
    time.sleep(0.5)  # Simulate API delay
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "conditions": "sunny"
    }

def get_news(location, category="general"):
    # In a real implementation, this would call a news API
    import time
    time.sleep(0.8)  # Simulate API delay
    return {
        "location": location,
        "headline": f"Major development announced in {location}",
        "source": "News API",
        "category": category
    }

# Define message
messages = [
    {"role": "user", "content": "What's the weather and news for New York and Los Angeles?"}
]

# Register functions
functions = {
    "get_weather": get_weather,
    "get_news": get_news
}

# Execute chat with functions in parallel
response = client.chat_with_functions(
    messages=messages,
    functions=functions,
    model="gpt-4o",
    max_rounds=3,
    function_timeout=5.0
)

print("Final response:", response.content)
```

### Metrics Collection for Distributed Testing

```python
# Metrics are automatically collected for all API calls

# Make some API calls
client.chat_completion(messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini")
client.embeddings("Test embedding", model="text-embedding-3-small")

# Access metrics
metrics = client.get_metrics()
print(f"Number of tracked requests: {len(metrics)}")

# Metrics include:
# - Request type (chat, embedding, tts, stt, etc.)
# - Start timestamp
# - Duration
# - Success status
# - Model used
# - Token counts (input, output, total)
# - Error type (if applicable)
# - Retry count
# - Request size
# - Response size

# Metrics are automatically reported to the distributed testing framework
# Export metrics to JSON
import json
with open("openai_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

## TypeScript Implementation

The TypeScript implementation provides the same features with a TypeScript-friendly interface:

### Core Features

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import { Message } from 'ipfs_accelerate_js/api_backends/types';

// Initialize the client
const client = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Chat with latest models
const messages: Message[] = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Hello, how are you?' }
];

// Use GPT-4o
const response = await client.chat(messages, {
  model: 'gpt-4o',
  temperature: 0.7
});

// Use more cost-effective GPT-4o-mini
const economicResponse = await client.chat(messages, {
  model: 'gpt-4o-mini',
  temperature: 0.7
});

// Get model information
console.log('GPT-4o max tokens:', client.getModelMaxTokens('gpt-4o'));
console.log('Does GPT-4o support vision?', client.supportsVision('gpt-4o'));
```

### Voice Features

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import { 
  OpenAIVoiceType, 
  OpenAIAudioFormat,
  OpenAITranscriptionFormat
} from 'ipfs_accelerate_js/api_backends/openai/types';
import * as fs from 'fs';

// Initialize the client
const client = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Enhanced Text-to-Speech with multiple voice options
const audioBuffer = await client.textToSpeech(
  'Hello, this is a demonstration of text-to-speech capabilities.',
  OpenAIVoiceType.NOVA,
  {
    model: 'tts-1-hd',
    responseFormat: OpenAIAudioFormat.MP3,
    speed: 1.1
  }
);

// Save the audio data
fs.writeFileSync('output.mp3', Buffer.from(audioBuffer));

// Speech-to-Text with word-level timestamps
const audioFile = fs.readFileSync('input.mp3');
const audioBlob = new Blob([audioFile], { type: 'audio/mp3' });

const transcription = await client.speechToText(audioBlob, {
  model: 'whisper-1',
  responseFormat: OpenAITranscriptionFormat.VERBOSE_JSON,
  timestamp_granularities: ['word']
});

// Access word-level timestamps
if (transcription.words && transcription.words.length > 0) {
  transcription.words.forEach(word => {
    console.log(`Word: ${word.word}, Start: ${word.start}s, End: ${word.end}s`);
  });
}

// Audio Translation
const nonEnglishAudio = fs.readFileSync('non_english.mp3');
const nonEnglishBlob = new Blob([nonEnglishAudio], { type: 'audio/mp3' });

const translation = await client.translateAudio(nonEnglishBlob, {
  model: 'whisper-1'
});
```

### Voice Agent

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import { OpenAIVoiceType, OpenAIAudioFormat } from 'ipfs_accelerate_js/api_backends/openai/types';
import * as fs from 'fs';

// Initialize the client
const client = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Create a voice agent for conversational interactions
const voiceAgent = client.createVoiceAgent(
  "You are a helpful assistant with expertise in technology.",
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

// Process text input and get spoken response
const textResult = await voiceAgent.processText(
  "What are the key differences between TypeScript and JavaScript?"
);

// Save the audio response
fs.writeFileSync('response.mp3', Buffer.from(textResult.audioResponse));
console.log('Text response:', textResult.textResponse);

// Process audio input
const audioFile = fs.readFileSync('question.mp3');
const audioBlob = new Blob([audioFile], { type: 'audio/mp3' });

const audioResult = await voiceAgent.processAudio(audioBlob);

// Access conversation history
const conversation = voiceAgent.getMessages();

// Reset the conversation
voiceAgent.reset("You are a technical support agent specializing in web development.");
```

### Parallel Function Calling

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';
import { Message } from 'ipfs_accelerate_js/api_backends/types';

// Initialize the client
const client = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Define messages
const messages: Message[] = [
  { role: 'user', content: 'What\'s the weather and news for New York and Los Angeles?' }
];

// Define functions to be executed in parallel
const functions = {
  get_weather: async (args: { location: string, unit?: string }) => {
    // In a real implementation, this would call a weather API
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
    return {
      location: args.location,
      temperature: args.unit === 'celsius' ? 22 : 72,
      conditions: 'sunny'
    };
  },
  get_news: async (args: { location: string, category?: string }) => {
    // In a real implementation, this would call a news API
    await new Promise(resolve => setTimeout(resolve, 800)); // Simulate API delay
    return {
      location: args.location,
      headline: `Major development announced in ${args.location}`,
      source: 'News API',
      category: args.category || 'general'
    };
  }
};

// Execute entire conversation with function calls in a single method
const response = await client.chatWithFunctions(messages, functions, {
  model: 'gpt-4o',
  temperature: 0.7,
  maxRounds: 3,             // Maximum rounds of function calling
  functionTimeout: 5000,    // Timeout for each function call in ms
  failOnFunctionError: false // Continue even if functions fail
});

console.log('Final response:', response.content);
```

### Metrics Collection for Distributed Testing

```typescript
import { OpenAI } from 'ipfs_accelerate_js/api_backends';

// Initialize the client
const client = new OpenAI({}, {
  openai_api_key: 'your-api-key'
});

// Metrics are automatically collected for all API calls

// Make some API calls
await client.chat([{ role: 'user', content: 'Hello' }], { model: 'gpt-4o-mini' });
await client.embedding('Test embedding', { model: 'text-embedding-3-small' });

// Access metrics
const metrics = (client as any).recentRequests;
console.log(`Number of tracked requests: ${Object.keys(metrics).length}`);

// Metrics include:
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
// Export metrics to JSON
const fs = require('fs');
fs.writeFileSync('openai_metrics.json', JSON.stringify(metrics, null, 2));
```

## Integration with Distributed Testing Framework

Both Python and TypeScript implementations are designed to work seamlessly with the IPFS Accelerate Distributed Testing Framework. Metrics collected during API calls are automatically reported to the framework for analysis and visualization.

### Key Metrics

- **Request Latency**: Time taken for each API call, broken down by request type and model
- **Token Usage**: Number of tokens used in requests and responses
- **Success Rate**: Percentage of successful API calls vs failures
- **Error Types**: Categorization of errors encountered (rate limits, authentication, etc.)
- **Retry Counts**: Number of retries needed for successful completion
- **Resource Usage**: Memory and CPU utilization during API calls

### Visualization and Analysis

The collected metrics are visualized in the Distributed Testing Dashboard, allowing for:

- Performance comparison across different models and API endpoints
- Trend analysis over time to identify performance degradation
- Anomaly detection to flag unusual behavior
- Cost analysis based on token usage and API calls
- Failure correlation to identify systematic issues

## Compatibility

Both implementations support the following OpenAI models and features:

### Chat Models
- GPT-4o (latest model with vision capabilities)
- GPT-4o-mini (efficient, economical alternative to GPT-4o)
- GPT-4-turbo, GPT-4
- GPT-3.5-turbo

### Embedding Models
- text-embedding-3-small (1536 dimensions)
- text-embedding-3-large (3072 dimensions, higher quality)
- text-embedding-ada-002 (legacy)

### Audio Models
- tts-1 (standard quality text-to-speech)
- tts-1-hd (high definition text-to-speech)
- whisper-1 (audio transcription and translation)

### Image Models
- dall-e-3

### Voice Options
- alloy (neutral voice)
- echo (male voice)
- fable (male storytelling voice)
- onyx (male authoritative voice)
- nova (female voice)
- shimmer (female soft voice)

## Future Enhancements

Planned enhancements for the next release (August 2025):

1. **Dynamic Model Selection**: Automatic model selection based on performance and cost constraints
2. **Streaming Voice Agent**: Real-time conversation with streaming responses
3. **Advanced Voice Control**: Voice-guided parameter selection and model configuration
4. **Adaptive Rate Limiting**: Dynamic adjustment of request rates based on API response patterns
5. **Contextual Embeddings Cache**: Smart caching of embeddings for frequently used contexts
6. **Cross-Implementation Testing**: Comprehensive test suite to verify consistency between Python and TypeScript implementations
7. **Advanced Model Analytics**: Detailed analysis of model performance characteristics and trade-offs