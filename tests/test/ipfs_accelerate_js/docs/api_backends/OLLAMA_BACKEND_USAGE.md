# Ollama Backend Usage Guide

This guide explains how to use the Ollama backend for working with local language models through the Ollama API.

## Introduction

The Ollama backend provides an interface for running large language models locally with [Ollama](https://ollama.ai/), giving you:

- Local LLM execution with no data sent to external servers
- Access to various open source models like Llama 3, Mistral, Gemma, Phi, and others
- Chat and streaming chat functionality
- Request queue with concurrency limits and circuit breaker pattern
- Detailed performance metrics and usage statistics

## Prerequisites

Before using the Ollama backend, you'll need:

1. Ollama installed and running on your system
2. At least one model pulled and available
3. Enough system resources to run LLMs locally

### Installing Ollama

Ollama is available for macOS, Linux, and Windows. Visit [ollama.ai](https://ollama.ai/) for installation instructions.

#### Basic Setup

```bash
# After installation, start the Ollama service
ollama serve

# In a separate terminal, pull a model (example: Llama 3)
ollama pull llama3
```

## Installation

The Ollama backend is included in the IPFS Accelerate JS package:

```bash
npm install ipfs-accelerate
```

Or if you're installing from source:

```bash
git clone https://github.com/your-org/ipfs-accelerate-js.git
cd ipfs-accelerate-js
npm install
```

## Basic Usage

### Initializing the Backend

```typescript
import { Ollama } from 'ipfs-accelerate/api_backends/ollama';

// By default, Ollama connects to http://localhost:11434/api
// You can override with environment variables or in metadata
const ollama = new Ollama(
  {}, // Resources (not needed for basic use)
  {
    // You can override the default URL
    ollama_api_url: process.env.OLLAMA_API_URL || 'http://localhost:11434/api',
    // Default model if not specified in requests
    model: process.env.OLLAMA_MODEL || 'llama3'
  }
);
```

Environment variables:
- `OLLAMA_API_URL`: Sets the API base URL (default: http://localhost:11434/api)
- `OLLAMA_MODEL`: Sets the default model (default: llama3)

### Basic Chat Generation

```typescript
import { Message } from 'ipfs-accelerate/api_backends/types';

// Define messages for the chat
const messages: Message[] = [
  { role: 'system', content: 'You are a helpful assistant that provides concise answers.' },
  { role: 'user', content: 'What is machine learning?' }
];

// Request options
const options = {
  temperature: 0.7,
  max_tokens: 150
};

// The model name to use ('llama3' is used as default if omitted)
const model = 'llama3'; // Change to a model you have pulled

// Get response
const response = await ollama.chat(model, messages, options);

console.log('Assistant:', response.text);

// You can access usage statistics
if (response.usage) {
  console.log('Usage statistics:');
  console.log(`- Prompt tokens: ${response.usage.prompt_tokens}`);
  console.log(`- Completion tokens: ${response.usage.completion_tokens}`);
  console.log(`- Total tokens: ${response.usage.total_tokens}`);
}
```

### Streaming Chat Responses

For real-time generation, you can use the streaming chat API:

```typescript
// Same message format as regular chat
const messages: Message[] = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Write a short poem about artificial intelligence.' }
];

// Same options format as regular chat
const options = {
  temperature: 0.8,
  max_tokens: 200
};

// Get streaming response
const streamingResponse = ollama.streamChat('llama3', messages, options);

// Process each chunk as it arrives
let fullResponse = '';

for await (const chunk of streamingResponse) {
  process.stdout.write(chunk.text); // Write incrementally to console
  fullResponse += chunk.text;
  
  if (chunk.done) {
    console.log('\nStreaming completed.');
  }
}

console.log('Full response:', fullResponse);
```

### Direct Text Generation

For simple text completions without chat history:

```typescript
// Simple prompt for text generation
const prompt = 'List three benefits of exercise:';

// Same options format as chat
const options = {
  temperature: 0.5,
  max_tokens: 100
};

const response = await ollama.generate('llama3', prompt, options);

console.log('Generated response:', response.text);
```

## Model Management

### Listing Available Models

```typescript
// Get all available models
const models = await ollama.listModels();

console.log('Available models:');
if (models.length === 0) {
  console.log('No models found. To pull a model, run: ollama pull llama3');
} else {
  models.forEach((model: any, index: number) => {
    console.log(`${index + 1}. ${model.name} (${model.size})`);
  });
}
```

### Testing Endpoint Availability

```typescript
// Check if the Ollama server is up and responding
const endpointWorking = await ollama.testOllamaEndpoint();

if (!endpointWorking) {
  console.log('Ollama endpoint is not responding. Is Ollama running?');
  console.log('Please start Ollama with: ollama serve');
}
```

### Model Compatibility Check

```typescript
// Check if a model is compatible with the Ollama backend
const modelName = 'llama3';
const isCompatible = ollama.isCompatibleModel(modelName);

console.log(`${modelName}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
```

## Advanced Features

### Configuration Parameters

The Ollama backend supports the following request parameters:

```typescript
// Full set of parameters
const options = {
  // Generation parameters
  temperature: 0.7,       // Controls randomness (0.0 to 1.0)
  max_tokens: 200,        // Maximum tokens to generate
  top_p: 0.9,             // Nucleus sampling parameter
  top_k: 40,              // Top-k sampling parameter
  
  // Internal parameters
  timeout: 60000,         // Request timeout in milliseconds
  priority: 'HIGH',       // Priority in the request queue
};
```

### Circuit Breaker Pattern

The Ollama backend implements a circuit breaker pattern for resilience:

```typescript
/*
 * The circuit breaker closes after a successful request and opens after
 * multiple failures (default: 5). When open, requests fail immediately.
 * After a timeout (default: 30 seconds), it enters half-open state and
 * allows a test request. Success closes the circuit again.
 *
 * This provides resilience by preventing cascading failures and
 * allowing the system to recover automatically.
 */

// The implementation is automatic, but you can see the stats
const backendInfo = ollama.getBackendInfo();
console.log(backendInfo);
```

### Usage Statistics

```typescript
// Reset usage statistics
ollama.resetUsageStats();

// Usage statistics are tracked automatically
// They are available in response objects
const response = await ollama.chat('llama3', messages);
console.log(response.usage);
```

## Error Handling

The Ollama backend includes robust error handling:

```typescript
try {
  const response = await ollama.chat('nonexistent-model', messages);
  console.log(response.text);
} catch (error) {
  // Handle errors appropriately
  console.error('Error:', error);
  
  // Check if it's a model not found error
  if (error.message.includes('model not found')) {
    console.log('You need to pull this model first with: ollama pull model-name');
  }
  
  // Check if it's a connection error
  if (error.message.includes('ECONNREFUSED')) {
    console.log('Ollama server is not running. Start it with: ollama serve');
  }
}
```

## Best Practices

### Model Selection

Choose appropriate models based on your needs:

1. **Smaller models** (7B parameter models like Phi-2, Gemma) for faster responses with limited resources
2. **Medium models** (Llama 3 8B, Mistral 7B) for good quality with reasonable performance
3. **Larger models** (Llama 70B, Mixtral) for highest quality output, but require more powerful hardware

```typescript
// For quick, lightweight use on modest hardware
const smallModel = 'phi-2';

// For general-purpose use with good quality
const mediumModel = 'llama3';

// For highest quality on powerful hardware
const largeModel = 'llama3:70b';
```

### System Requirements

Ollama model requirements vary:

- **7B models**: 8GB RAM minimum, 16GB recommended
- **13B models**: 16GB RAM minimum, 32GB recommended
- **70B models**: 32GB RAM minimum, 64GB recommended, GPU strongly recommended

### Performance Optimization

```typescript
// For faster responses with less randomness (deterministic)
const fastOptions = {
  temperature: 0.1,
  max_tokens: 50
};

// For more creative, diverse responses
const creativeOptions = {
  temperature: 0.9,
  top_p: 0.92,
  max_tokens: 200
};
```

### Environment Setup

Using environment variables for configuration:

```bash
# Set in your environment
export OLLAMA_API_URL="http://localhost:11434/api"
export OLLAMA_MODEL="llama3"
```

## Examples

### Complete Chat Example

```typescript
import { Ollama } from 'ipfs-accelerate/api_backends/ollama';
import { Message } from 'ipfs-accelerate/api_backends/types';

async function chatExample() {
  try {
    // Initialize backend
    const ollama = new Ollama(
      {},
      { 
        ollama_api_url: process.env.OLLAMA_API_URL || 'http://localhost:11434/api',
        model: 'llama3'
      }
    );
    
    // Check if the endpoint is working
    const isWorking = await ollama.testOllamaEndpoint();
    
    if (!isWorking) {
      console.log('Ollama endpoint is not available. Please check if Ollama is running.');
      return;
    }
    
    // Prepare a conversation
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant who is knowledgeable about AI.' },
      { role: 'user', content: 'What is the difference between supervised and unsupervised learning?' }
    ];
    
    // Generate response
    console.log('Generating response...');
    
    const response = await ollama.chat('llama3', messages, {
      temperature: 0.7,
      max_tokens: 200
    });
    
    console.log('Question: What is the difference between supervised and unsupervised learning?');
    console.log('Response:', response.text);
    
    if (response.usage) {
      console.log(`Total tokens: ${response.usage.total_tokens}`);
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

chatExample();
```

### Streaming Generation Example

```typescript
import { Ollama } from 'ipfs-accelerate/api_backends/ollama';
import { Message } from 'ipfs-accelerate/api_backends/types';

async function streamingExample() {
  try {
    // Initialize backend
    const ollama = new Ollama();
    
    // Prepare a conversation
    const messages: Message[] = [
      { role: 'system', content: 'You are a creative assistant who writes poetry.' },
      { role: 'user', content: 'Write a short haiku about programming.' }
    ];
    
    console.log('Question: Write a short haiku about programming.');
    console.log('Streaming response:');
    
    // Stream the response
    const stream = ollama.streamChat('llama3', messages, {
      temperature: 0.8,
      max_tokens: 100
    });
    
    // Process each chunk
    for await (const chunk of stream) {
      process.stdout.write(chunk.text);
      
      if (chunk.done) {
        console.log('\nStream completed.');
      }
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

streamingExample();
```

## API Reference

### Constructor

```typescript
new Ollama(resources?: Record<string, any>, metadata?: ApiMetadata)
```

**Parameters:**
- `resources`: Optional resources to pass to the backend
- `metadata`: Configuration options including:
  - `ollama_api_url`: Ollama API base URL (default: "http://localhost:11434/api")
  - `ollama_model` or `model`: Default model name
  - Other standard ApiMetadata properties

### Methods

#### `chat(model: string, messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse>`

Generate a chat completion from a list of messages.

**Parameters:**
- `model`: Model name (e.g., "llama3")
- `messages`: Array of message objects with role and content
- `options`: Optional parameters including:
  - `temperature`: Sampling temperature (0.0 to 1.0)
  - `max_tokens`: Maximum tokens to generate
  - `top_p`: Nucleus sampling parameter
  - `top_k`: Top-k sampling parameter

**Returns:** Chat completion response with text and usage statistics.

#### `streamChat(model: string, messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Generate a streaming chat completion.

**Parameters:** Same as `chat` method.

**Returns:** Async generator yielding stream chunks with partial text.

#### `generate(model: string, prompt: string, options?: ApiRequestOptions): Promise<ChatCompletionResponse>`

Generate a text completion from a prompt.

**Parameters:**
- `model`: Model name (e.g., "llama3")
- `prompt`: Text prompt
- `options`: Same options as `chat` method

**Returns:** Chat completion response with text and usage statistics.

#### `listModels(): Promise<any[]>`

List available models in Ollama.

**Returns:** Array of model objects with name and size information.

#### `testOllamaEndpoint(endpointUrl?: string): Promise<boolean>`

Test if the Ollama endpoint is available.

**Parameters:**
- `endpointUrl`: Optional endpoint URL (defaults to configured URL)

**Returns:** Boolean indicating if the endpoint is working.

#### `resetUsageStats(): void`

Reset usage statistics to zero.

#### `isCompatibleModel(modelName: string): boolean`

Check if a model is compatible with this backend.

**Parameters:**
- `modelName`: Name of the model to check

**Returns:** Boolean indicating if the model is compatible.

#### `getBackendInfo(): Record<string, any>`

Get information about the backend implementation.

**Returns:** Object with backend details and capabilities.

## Compatibility

The Ollama backend is compatible with:

- Ollama version 0.1.0 and later
- Various Ollama-supported models, including:
  - `llama3`: Meta's Llama 3 models
  - `llama2`: Meta's Llama 2 models
  - `mistral`: Mistral AI's models
  - `phi`: Microsoft's Phi models
  - `gemma`: Google's Gemma models
  - `codellama`: Code specialized Llama models
  - Any custom model pulled into Ollama

## Additional Resources

- [Ollama Official Website](https://ollama.ai/)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Model Library](https://ollama.ai/library)
- [IPFS Accelerate JS Documentation](https://github.com/your-org/ipfs-accelerate-js)