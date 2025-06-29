# HuggingFace Text Generation Inference Unified Backend Usage Guide

This guide explains how to use the HuggingFace Text Generation Inference (HF TGI) Unified backend for generating text with both the HuggingFace Inference API and self-hosted container deployments.

## Introduction

The HF TGI Unified backend provides a versatile interface for working with text generation models:

- **Dual-mode operation**: Works with both the hosted HuggingFace API and container-based deployments
- **Streaming text generation**: Real-time incremental text output
- **Chat interface**: Support for conversation history and various chat formats
- **Container management**: Built-in Docker container deployment and management
- **Performance benchmarking**: Tools for evaluating text generation performance
- **Advanced prompt engineering**: Multiple prompt template formats for different model families
- **Circuit breaker pattern**: Prevents cascading failures by stopping API requests after consecutive errors
- **Request queue management**: Handles rate limiting and prioritizes requests
- **Robust error handling**: Comprehensive error classification with typed error handling
- **Advanced use cases**: Ready-to-use patterns for QA systems, conversation agents, and document summarization

## Prerequisites

Before using the HF TGI Unified backend, you'll need:

1. For API mode:
   - A HuggingFace account
   - A HuggingFace API key (for accessing hosted models)

2. For container mode:
   - Docker installed on your system
   - Sufficient disk space for model downloads (2-7GB per model)
   - NVIDIA GPU with CUDA support (strongly recommended for larger models)

### Obtaining an API Key

1. Create an account on [HuggingFace](https://huggingface.co/join)
2. Navigate to [Settings > API Tokens](https://huggingface.co/settings/tokens)
3. Create a new API token
4. Save the token securely

## Installation

The HF TGI Unified backend is included in the IPFS Accelerate JS package:

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
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';
import { HfTgiUnifiedOptions, HfTgiUnifiedApiMetadata } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

// Set up configuration options
const options: HfTgiUnifiedOptions = {
  apiUrl: 'https://api-inference.huggingface.co/models',
  maxRetries: 3,
  requestTimeout: 60000,
  useRequestQueue: true,
  debug: false,
  // Default generation parameters
  maxTokens: 100,
  temperature: 0.7,
  topP: 0.95,
  topK: 50,
  repetitionPenalty: 1.1
};

// Set up metadata with API key
const metadata: HfTgiUnifiedApiMetadata = {
  hf_api_key: 'YOUR_API_KEY', // Can also use process.env.HF_API_KEY
  model_id: 'google/flan-t5-small' // Default model to use
};

// Create the backend instance
const hfTgiUnified = new HfTgiUnified(options, metadata);
```

You can also provide the API key as an environment variable:

```bash
export HF_API_KEY=your_api_key
```

### Basic Text Generation

```typescript
// Generate text with default parameters
const prompt = "Write a short poem about artificial intelligence.";
const generatedText = await hfTgiUnified.generateText(prompt);
console.log('Generated text:', generatedText);

// Generate text with custom parameters
const options = {
  model: 'google/flan-t5-small',
  maxTokens: 150,
  temperature: 0.8,
  topP: 0.95,
  topK: 50,
  repetitionPenalty: 1.2,
  priority: 'HIGH'
};
const customText = await hfTgiUnified.generateText(prompt, options);
console.log('Generated text with custom parameters:', customText);
```

### Streaming Text Generation

```typescript
// Stream text generation for real-time output
const prompt = "List 5 benefits of exercise.";
const options = {
  model: 'google/flan-t5-small',
  maxTokens: 150,
  temperature: 0.7,
  stream: true
};

console.log('Prompt:', prompt);
console.log('Streaming response:');

let fullText = '';
const streamGenerator = hfTgiUnified.streamGenerateText(prompt, options);

for await (const chunk of streamGenerator) {
  process.stdout.write(chunk.text);
  fullText += chunk.text;
  
  if (chunk.done) {
    console.log('\nStreaming completed.');
  }
}

console.log('Full generated text:', fullText);
```

## Chat API

The HF TGI Unified backend includes a chat interface that formats messages appropriately for text generation models:

### Basic Chat Completion

```typescript
import { ChatMessage } from 'ipfs-accelerate/api_backends/types';
import { ChatGenerationOptions } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

// Define a conversation with multiple messages
const messages: ChatMessage[] = [
  { role: 'system', content: 'You are a helpful AI assistant specialized in science.' },
  { role: 'user', content: 'What are black holes?' },
  { role: 'assistant', content: 'Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape from them.' },
  { role: 'user', content: 'How are they formed?' }
];

const chatOptions: ChatGenerationOptions = {
  model: 'google/flan-t5-small',
  maxTokens: 150,
  temperature: 0.7,
  systemMessage: 'You are a helpful AI assistant specialized in science.',
  promptTemplate: 'chat'
};

const response = await hfTgiUnified.chat(messages, chatOptions);
console.log('Chat response:', response.text);
```

### Streaming Chat Completion

```typescript
// Define a simple conversation
const messages: ChatMessage[] = [
  { role: 'system', content: 'You are a helpful AI assistant.' },
  { role: 'user', content: 'Tell me a short story about a robot learning to paint.' }
];

const chatOptions: ChatGenerationOptions = {
  model: 'google/flan-t5-small',
  maxTokens: 200,
  temperature: 0.8,
  promptTemplate: 'instruction'
};

console.log('Streaming chat response:');

const chatStream = hfTgiUnified.streamChat(messages, chatOptions);

for await (const chunk of chatStream) {
  process.stdout.write(chunk.text);
  
  if (chunk.done) {
    console.log('\nChat streaming completed.');
  }
}
```

### Prompt Templates

The HF TGI Unified backend supports different prompt templates for various model families:

```typescript
// Choose a prompt template based on the model
const chatOptions: ChatGenerationOptions = {
  promptTemplate: 'llama2', // For Llama 2 models
  maxTokens: 150
};

// Available prompt templates:
// 'default': "{input_text}"
// 'instruction': "### Instruction:\n{input_text}\n\n### Response:"
// 'chat': "{system_message}\n\n{messages}"
// 'llama2': "<s>[INST] {input_text} [/INST]"
// 'falcon': "User: {input_text}\n\nAssistant:"
// 'mistral': "<s>[INST] {input_text} [/INST]"
// 'chatml': "<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
```

## Container Mode

The HF TGI Unified backend can manage Docker containers running the HuggingFace Text Generation Inference server.

### Switching to Container Mode

```typescript
// Switch to container mode
hfTgiUnified.setMode(true);
console.log(`Current mode: ${hfTgiUnified.getMode()}`); // Should output 'container'
```

### Starting a Container

```typescript
import { DeploymentConfig } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

// Define container deployment configuration
const deployConfig: DeploymentConfig = {
  dockerRegistry: 'ghcr.io/huggingface/text-generation-inference',
  containerTag: 'latest',
  gpuDevice: '0', // GPU device ID, use empty string for CPU-only: ''
  modelId: 'google/flan-t5-small',
  port: 8080,
  env: {
    'HF_API_TOKEN': 'your_api_key' // Optional for accessing gated models
  },
  volumes: ['./cache:/data'], // Optional volume mounts
  network: 'bridge',
  maxInputLength: 2048, // Optional parameters
  parameters: ['--max-batch-size=32'] // Optional additional parameters
};

// Start the container
const containerInfo = await hfTgiUnified.startContainer(deployConfig);

console.log('Container started:');
console.log(`- Container ID: ${containerInfo.containerId}`);
console.log(`- Host: ${containerInfo.host}`);
console.log(`- Port: ${containerInfo.port}`);
console.log(`- Status: ${containerInfo.status}`);

// Generate text using the container
const prompt = "Generate a creative idea for a mobile app.";
const generatedText = await hfTgiUnified.generateText(prompt);

console.log('Generated text using container:', generatedText);
```

### Stopping the Container

```typescript
// Stop and remove the container when done
const stopped = await hfTgiUnified.stopContainer();
console.log(`Container stopped: ${stopped}`);
```

## Model Information and Testing

### Getting Model Information

```typescript
const modelInfo = await hfTgiUnified.getModelInfo();

console.log('Model information:');
console.log(`- Model ID: ${modelInfo.model_id}`);
console.log(`- Status: ${modelInfo.status}`);
console.log(`- Revision: ${modelInfo.revision}`);
console.log(`- Framework: ${modelInfo.framework}`);
console.log(`- Max input length: ${modelInfo.max_input_length}`);
console.log(`- Max total tokens: ${modelInfo.max_total_tokens}`);
console.log(`- Parameters: ${modelInfo.parameters.join(', ')}`);
```

### Testing Endpoint Availability

```typescript
const isAvailable = await hfTgiUnified.testEndpoint();
console.log(`Endpoint available: ${isAvailable}`);
```

### Model Compatibility

```typescript
// Check if models are compatible with the backend
const models = [
  'google/flan-t5-small',
  'facebook/opt-125m',
  'bigscience/bloom-560m',
  'tiiuae/falcon-7b',
  'mistralai/Mistral-7B-v0.1',
  'random-model-name'
];

for (const model of models) {
  const isCompatible = hfTgiUnified.isCompatibleModel(model);
  console.log(`${model}: ${isCompatible ? 'Compatible' : 'Not compatible'}`);
}
```

## Performance Benchmarking

The backend includes tools for benchmarking text generation performance:

```typescript
const benchmarkOptions = {
  iterations: 5, // Number of iterations to run
  model: 'google/flan-t5-small',
  maxTokens: 100
};

const benchmarkResults = await hfTgiUnified.runBenchmark(benchmarkOptions);

console.log('Benchmark results:');
console.log(`- Single generation time: ${benchmarkResults.singleGenerationTime.toFixed(2)} ms`);
console.log(`- Tokens per second: ${benchmarkResults.tokensPerSecond.toFixed(2)}`);
console.log(`- Generated tokens: ${benchmarkResults.generatedTokens.toFixed(2)}`);
console.log(`- Input tokens: ${benchmarkResults.inputTokens}`);
```

## Error Handling

The HF TGI Unified backend includes comprehensive error handling:

```typescript
try {
  const text = await hfTgiUnified.generateText("Test prompt");
  console.log(`Generated text: ${text}`);
} catch (error) {
  if (error.message.includes('not found')) {
    console.error('Model not found. Check the model ID and try again.');
  } else if (error.message.includes('Authorization')) {
    console.error('API key error. Check your API key and permissions.');
  } else if (error.message.includes('loading')) {
    console.error('Model is still loading. Try again in a few moments.');
  } else {
    console.error('Error generating text:', error);
  }
}
```

Common error scenarios:
- Invalid API key
- Model not found
- Rate limiting
- Server errors
- Model loading delays
- Container startup issues

### Robust Error Handling Pattern

For production applications, we recommend implementing a robust error handling pattern with typed error classification and exponential backoff:

```typescript
// Robust error handling with exponential backoff
const robustGenerateText = async (prompt: string, options: TextGenerationOptions = {}, retries = 3): Promise<string> => {
  try {
    return await hfTgiUnified.generateText(prompt, options);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    
    // Check for retriable errors
    if (
      errorMessage.includes('loading') || 
      errorMessage.includes('busy') || 
      errorMessage.includes('rate limit') ||
      errorMessage.includes('timeout')
    ) {
      if (retries > 0) {
        // Exponential backoff
        const delay = 1000 * Math.pow(2, 3 - retries);
        console.log(`Retrying after ${delay}ms (${retries} retries left)...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        return robustGenerateText(prompt, options, retries - 1);
      }
    }
    
    // Re-throw non-retriable errors or when out of retries
    throw error;
  }
};

// Usage:
try {
  const text = await robustGenerateText("Test prompt", { temperature: 0.7 });
  console.log("Generated text:", text);
} catch (error) {
  console.error("Failed after multiple retries:", error);
}
```

### Circuit Breaker Pattern

The HF TGI Unified backend includes a built-in circuit breaker pattern that helps prevent cascading failures:

- Tracks consecutive failures and temporarily disables requests after a threshold is reached
- Automatically resets after a cooldown period
- Configurable failure threshold and reset timeout
- Used internally by the backend for API resilience

The circuit breaker configuration includes:
- Failure threshold: 3 consecutive failures
- Reset timeout: 30000ms (30 seconds)

This feature is particularly useful for high-volume applications or when working with potentially unstable endpoints.

## Best Practices

### Environment Variables

Store your API key in environment variables:

```typescript
const apiKey = process.env.HF_API_KEY;
const hfTgiUnified = new HfTgiUnified({}, { hf_api_key: apiKey });
```

### Model Selection

Choose appropriate models for your use case:

1. **Small models** (`t5-small`, `opt-125m`) for API mode with quick responses
2. **Medium models** (`flan-t5-base`, `bloom-560m`) for better quality with reasonable performance
3. **Large models** (`llama-2-7b`, `falcon-7b`) for high-quality output, but require GPU in container mode

```typescript
// For quick, lightweight use
const smallModel = new HfTgiUnified({}, { 
  hf_api_key: apiKey, 
  model_id: 'google/flan-t5-small' 
});

// For high-quality in container mode
const largeModel = new HfTgiUnified({
  useContainer: true
}, { 
  model_id: 'meta-llama/Llama-2-7b-chat-hf' 
});
```

### Container Resources

When using container mode:

1. Ensure sufficient GPU memory for the model
2. Consider setting volume mounts for caching downloaded models
3. Monitor container resource usage for production deployments
4. Stop containers when not in use to free resources

### Generation Parameters

Adjust generation parameters based on your needs:

```typescript
// For creative, diverse text
const creativeOptions = {
  temperature: 0.8, // Higher for more randomness
  topP: 0.92,
  topK: 50,
  maxTokens: 200
};

// For factual, deterministic text
const factualOptions = {
  temperature: 0.1, // Lower for more deterministic
  topP: 0.95,
  repetitionPenalty: 1.2,
  maxTokens: 100
};
```

## Compatibility

The HF TGI Unified backend is compatible with:

- A wide range of text generation models on HuggingFace Hub
- Docker containers for self-hosted deployments
- CPU and GPU hardware acceleration (GPU strongly recommended for container mode)

Popular compatible models:
- `google/flan-t5-small`: Small, efficient instruction-following model
- `google/flan-t5-base`: Base-sized instruction-following model
- `facebook/opt-125m`: Small, general-purpose text generation model
- `facebook/opt-350m`: Medium-sized text generation model
- `bigscience/bloom-560m`: Multilingual text generation model
- `tiiuae/falcon-7b`: High-quality text generation model (requires GPU)
- `meta-llama/Llama-2-7b-hf`: General text generation model (requires GPU)
- `meta-llama/Llama-2-7b-chat-hf`: Conversation-tuned model (requires GPU)
- `mistralai/Mistral-7B-v0.1`: High-quality text generation model (requires GPU)
- `mistralai/Mistral-7B-Instruct-v0.1`: Instruction-tuned model (requires GPU)
- `microsoft/phi-1_5`: Efficient small language model
- `microsoft/phi-2`: Efficient small language model with good performance

## Advanced Use Cases

### Question Answering System

Build a simple question answering system that can extract information from provided context:

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';
import { TextGenerationOptions } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

async function buildQASystem() {
  const backend = new HfTgiUnified({
    maxTokens: 100,
    temperature: 0.3 // Lower temperature for more factual answers
  }, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'google/flan-t5-small'
  });
  
  const answerQuestion = async (context: string, question: string): Promise<string> => {
    const prompt = `
Context: ${context}

Question: ${question}

Answer:
`;
    
    const options: TextGenerationOptions = {
      maxTokens: 100,
      temperature: 0.3,
      topP: 0.95,
      repetitionPenalty: 1.2
    };
    
    return await backend.generateText(prompt, options);
  };
  
  // Example usage
  const context = `
The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. 
The U.S. National Aeronautics and Space Administration (NASA) led development of the telescope in collaboration 
with the European Space Agency (ESA) and the Canadian Space Agency (CSA). The telescope is named after James E. Webb, 
who was the administrator of NASA from 1961 to 1968 during the Mercury, Gemini, and Apollo programs.
The telescope was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana, 
and arrived at the Sun–Earth L2 Lagrange point in January 2022.
`;
  const question = "When was the James Webb Space Telescope launched?";
  
  console.log(`Question: ${question}`);
  const answer = await answerQuestion(context, question);
  console.log(`Answer: ${answer}`);
}

buildQASystem().catch(console.error);
```

### Multi-Turn Conversation Agent

Create a stateful conversation agent with persistent conversation history:

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';
import { ChatMessage } from 'ipfs-accelerate/api_backends/types';
import { ChatGenerationOptions } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

async function createConversationAgent() {
  // Initialize backend
  const backend = new HfTgiUnified({
    maxTokens: 150,
    temperature: 0.7
  }, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'google/flan-t5-small'
  });
  
  interface ConversationAgent {
    name: string;
    persona: string;
    conversation: ChatMessage[];
    addUserMessage: (message: string) => void;
    getResponse: () => Promise<string>;
    getConversationHistory: () => ChatMessage[];
  }
  
  // Create a conversation agent
  const createAgent = (name: string, persona: string): ConversationAgent => {
    const agent: ConversationAgent = {
      name,
      persona,
      conversation: [
        { role: 'system', content: persona }
      ],
      addUserMessage(message: string) {
        this.conversation.push({ role: 'user', content: message });
      },
      async getResponse() {
        // Generate a response from the model
        const chatOptions: ChatGenerationOptions = {
          maxTokens: 150,
          temperature: 0.7,
          topP: 0.9,
          systemMessage: this.persona
        };
        
        const response = await backend.chat(this.conversation, chatOptions);
        
        // Add the response to the conversation history
        this.conversation.push({ role: 'assistant', content: response.text });
        
        return response.text;
      },
      getConversationHistory() {
        return this.conversation;
      }
    };
    
    return agent;
  };
  
  // Example usage
  const travelAgent = createAgent(
    'TravelBot',
    'You are a helpful travel assistant. You provide concise and specific travel recommendations based on user preferences.'
  );
  
  // First user message
  travelAgent.addUserMessage("I want to visit a warm place with beaches in December.");
  console.log('User: I want to visit a warm place with beaches in December.');
  
  // Get first response
  const response1 = await travelAgent.getResponse();
  console.log(`${travelAgent.name}: ${response1}`);
  
  // Second user message
  travelAgent.addUserMessage("I prefer places where English is commonly spoken. What do you recommend?");
  console.log('User: I prefer places where English is commonly spoken. What do you recommend?');
  
  // Get second response
  const response2 = await travelAgent.getResponse();
  console.log(`${travelAgent.name}: ${response2}`);
}

createConversationAgent().catch(console.error);
```

### Document Summarization

Create a document summarization utility that condenses text while preserving key information:

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';
import { TextGenerationOptions } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

async function summarizeDocuments() {
  const backend = new HfTgiUnified({
    maxTokens: 200,
    temperature: 0.5
  }, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'google/flan-t5-small'
  });
  
  const summarizeDocument = async (document: string, maxLength: number = 150): Promise<string> => {
    const prompt = `
Summarize the following document in a concise way, highlighting the key points. 
Keep the summary to around ${maxLength} words.

Document:
${document}

Summary:
`;
    
    const options: TextGenerationOptions = {
      maxTokens: maxLength * 2, // Allow enough tokens for a good summary
      temperature: 0.5,
      topP: 0.95
    };
    
    return await backend.generateText(prompt, options);
  };
  
  // Example usage
  const document = `
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals including humans. 
AI applications include advanced web search engines, recommendation systems, language translation, autonomous driving, and creating art.

Machine learning is a subset of AI where computers can learn and improve from experience without being explicitly programmed. 
In recent years, deep learning, a form of machine learning based on artificial neural networks, has led to significant breakthroughs 
in image recognition, natural language processing, and reinforcement learning.

The field of AI raises serious ethical concerns. The development of superintelligent AI systems could pose existential risks, 
while increasing automation may lead to significant economic disruption. Other AI safety considerations include preventing AI 
from being programmed for destructive uses like autonomous weapons, ensuring AI systems don't develop harmful emergent behaviors, 
and addressing issues of AI alignment with human values.

Despite these concerns, AI has the potential to solve many of humanity's most pressing challenges, from climate change to disease. 
The responsible development of AI systems represents one of the most important technological frontiers of our time.
`;
  
  console.log('Original document length:', document.split(/\s+/).length, 'words');
  const summary = await summarizeDocument(document.trim(), 75);
  console.log(`Summary: ${summary}`);
  console.log('Summary length:', summary.split(/\s+/).length, 'words');
}

summarizeDocuments().catch(console.error);
```

## Examples

### Complete Basic Example

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';

async function generateText() {
  // Initialize with API key from environment
  const backend = new HfTgiUnified({
    maxTokens: 100,
    temperature: 0.7
  }, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'google/flan-t5-small'
  });
  
  // Test if the endpoint is available
  const isAvailable = await backend.testEndpoint();
  console.log(`Endpoint available: ${isAvailable}`);
  
  if (isAvailable) {
    try {
      // Generate text
      const prompt = "Write a short story about a time traveler visiting ancient Rome.";
      const generatedText = await backend.generateText(prompt);
      
      console.log('Prompt:', prompt);
      console.log('Generated text:', generatedText);
      
      return generatedText;
    } catch (error) {
      console.error('Error generating text:', error);
    }
  } else {
    console.log('Endpoint not available. Please check your API key and connection.');
  }
}

generateText().catch(console.error);
```

### Streaming Chat Example

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';
import { ChatMessage } from 'ipfs-accelerate/api_backends/types';
import { ChatGenerationOptions } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

async function streamingChat() {
  // Initialize backend
  const backend = new HfTgiUnified({
    maxTokens: 200,
    temperature: 0.7
  }, {
    hf_api_key: process.env.HF_API_KEY,
    model_id: 'google/flan-t5-small'
  });
  
  // Define conversation
  const messages: ChatMessage[] = [
    { role: 'system', content: 'You are a helpful AI assistant who specializes in explaining complex topics simply.' },
    { role: 'user', content: 'Explain how nuclear fusion works in simple terms.' }
  ];
  
  const chatOptions: ChatGenerationOptions = {
    maxTokens: 300,
    temperature: 0.7,
    promptTemplate: 'instruction'
  };
  
  console.log('Question: Explain how nuclear fusion works in simple terms.');
  console.log('Streaming response:');
  
  try {
    // Stream the chat response
    const chatStream = backend.streamChat(messages, chatOptions);
    
    for await (const chunk of chatStream) {
      process.stdout.write(chunk.text);
      
      if (chunk.done) {
        console.log('\nChat streaming completed.');
      }
    }
  } catch (error) {
    console.error('Error in streaming chat:', error);
  }
}

streamingChat().catch(console.error);
```

### Container Deployment Example

```typescript
import { HfTgiUnified } from 'ipfs-accelerate/api_backends/hf_tgi_unified';
import { DeploymentConfig } from 'ipfs-accelerate/api_backends/hf_tgi_unified/types';

async function runTextGenerationContainer() {
  try {
    // Initialize backend in container mode
    const backend = new HfTgiUnified({
      useContainer: true,
      containerUrl: 'http://localhost:8080',
      debug: true
    }, {
      model_id: 'facebook/opt-125m' // Small model for example purposes
    });
    
    // Define container configuration
    const config: DeploymentConfig = {
      dockerRegistry: 'ghcr.io/huggingface/text-generation-inference',
      containerTag: 'latest',
      gpuDevice: '0', // Use GPU device 0
      modelId: 'facebook/opt-125m',
      port: 8080,
      env: {},
      volumes: ['./cache:/data'],
      network: 'bridge',
      maxInputLength: 1024
    };
    
    // Start the container
    console.log('Starting container...');
    const containerInfo = await backend.startContainer(config);
    console.log('Container started:', containerInfo);
    
    // Generate text using the container
    const prompt = "Once upon a time in a magical forest,";
    const generatedText = await backend.generateText(prompt, {
      maxTokens: 100,
      temperature: 0.8
    });
    
    console.log('Prompt:', prompt);
    console.log('Generated text:', generatedText);
    
    // Stop the container
    console.log('Stopping container...');
    await backend.stopContainer();
    console.log('Container stopped');
  } catch (error) {
    console.error('Error:', error);
  }
}

runTextGenerationContainer().catch(console.error);
```

## Performance Benchmarking and Optimization

The HF TGI Unified backend includes a built-in benchmarking system to help you measure and optimize text generation performance:

### Basic Benchmarking

```typescript
// Set up benchmark options
const benchmarkOptions = {
  iterations: 5,           // Number of iterations for reliable results
  model: 'google/flan-t5-small',  // Model to benchmark
  maxTokens: 50,           // Maximum tokens to generate
  temperature: 0.7,        // Temperature parameter
  topP: 0.95               // Top-p sampling parameter
};

// Run the benchmark
const benchmarkResults = await hfTgiUnified.runBenchmark(benchmarkOptions);

// Output results
console.log('Benchmark results:');
console.log(`- Single generation time: ${benchmarkResults.singleGenerationTime.toFixed(2)} ms`);
console.log(`- Tokens per second: ${benchmarkResults.tokensPerSecond.toFixed(2)}`);
console.log(`- Generated tokens per request: ${benchmarkResults.generatedTokens.toFixed(2)}`);
console.log(`- Input tokens per request: ${benchmarkResults.inputTokens}`);
```

### Performance Optimization Strategies

The backend includes several features to help optimize performance:

1. **Smaller models for lower latency**: Use smaller models like `google/flan-t5-small` or `facebook/opt-125m` when response time is critical.

2. **Container mode for high throughput**: Deploy models in container mode to avoid API overhead and rate limits:
   ```typescript
   // Initialize in container mode for better performance
   const backend = new HfTgiUnified({
     useContainer: true,
     containerUrl: 'http://localhost:8080'
   }, { model_id: 'google/flan-t5-small' });
   ```

3. **Parameter optimization**: Fine-tune generation parameters for your specific use case:
   ```typescript
   // For factual, deterministic tasks (faster)
   const factualOptions = {
     temperature: 0.1,      // Lower temperature for more deterministic output
     topP: 0.95,            // Standard top-p
     repetitionPenalty: 1.2 // Prevent repetition
   };
   
   // For creative tasks (may be slower)
   const creativeOptions = {
     temperature: 0.8,      // Higher temperature for more variety
     topP: 0.92,            // Slightly more diverse sampling
     topK: 50,              // Consider top 50 tokens
     repetitionPenalty: 1.1 // Light repetition penalty
   };
   ```

4. **Priority levels for important requests**: Set priority levels for critical requests:
   ```typescript
   // High priority for critical requests
   const criticalOptions = {
     priority: 'HIGH',
     maxTokens: 100
   };
   
   // Low priority for background tasks
   const backgroundOptions = {
     priority: 'LOW',
     maxTokens: 200
   };
   ```

5. **Stop sequences for efficient generation**: Use stop sequences to end generation early:
   ```typescript
   // Stop generation after specific text is generated
   const options = {
     stopSequences: ['5.', '5)', '5:'] // Stop after item 5
   };
   ```

## Additional Resources

- [HuggingFace Inference API Documentation](https://huggingface.co/docs/api-inference/index)
- [Text Generation Inference GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [HuggingFace Text Generation Models](https://huggingface.co/models?pipeline_tag=text-generation)
- [IPFS Accelerate JS Documentation](https://github.com/your-org/ipfs-accelerate-js)
- [Docker Installation Guide](https://docs.docker.com/get-docker/)
- [IPFS Accelerate TypeScript SDK Guide](../../../IPFS_ACCELERATE_JS_IMPLEMENTATION_SUMMARY.md)
- [API Backends TypeScript Migration Guide](../../../API_BACKENDS_TYPESCRIPT_MIGRATION_PLAN.md)