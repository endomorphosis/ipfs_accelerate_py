# API Backend Development Guide

This guide explains how to create new API backends for the IPFS Accelerate JavaScript SDK.

## Overview

The IPFS Accelerate JavaScript SDK uses a modular API backend system to provide a consistent interface for different AI service providers. Each API backend implements a common interface that provides:

- Authentication and API key management
- Request handling and formatting
- Error handling and retries
- Streaming support
- Circuit breaker pattern
- Resource pooling

## Getting Started

To create a new API backend, start by duplicating the sample_backend directory:

```bash
cp -r src/api_backends/sample_backend src/api_backends/your_backend_name
```

Rename the files to match your backend name:

```bash
mv src/api_backends/your_backend_name/sample_backend.ts src/api_backends/your_backend_name/your_backend_name.ts
```

Update the imports and exports in the index.ts file to reference your backend.

## Required Components

### 1. Type Definitions

Create type definitions in `types.ts` for your backend's request and response formats. For example:

```typescript
import { Message } from '../types';

export interface YourBackendResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message?: {
      role: string;
      content: string;
    };
    delta?: {
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface YourBackendRequest {
  model: string;
  messages?: Message[];
  prompt?: string;
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  // Add any backend-specific parameters
}
```

### 2. Backend Class Implementation

Create your backend class by extending `BaseApiBackend` in your main file. Implement the required methods:

```typescript
import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import { YourBackendResponse, YourBackendRequest } from './types';

export class YourBackend extends BaseApiBackend {
  private apiEndpoint: string = 'https://api.yourbackend.com/v1/chat';
  private defaultModel: string = 'your-default-model';

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }

  // Required methods to implement
  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.your_backend_api_key || 
           metadata.yourBackendApiKey || 
           (typeof process !== 'undefined' ? process.env.YOUR_BACKEND_API_KEY || '' : '');
  }

  protected getDefaultModel(): string {
    return this.metadata.model as string || this.defaultModel;
  }

  isCompatibleModel(model: string): boolean {
    // Return true if this model is compatible with this backend
    return model.toLowerCase().includes('your_backend');
  }

  createEndpointHandler(): (data: any) => Promise<any> {
    // Implementation here
  }

  async testEndpoint(): Promise<boolean> {
    // Implementation here
  }

  async chat(messages: Message[], options: ApiRequestOptions = {}): Promise<ChatCompletionResponse> {
    // Implementation here
  }

  async *streamChat(messages: Message[], options: ApiRequestOptions = {}): AsyncGenerator<StreamChunk> {
    // Implementation here
  }
}
```

### 3. Request Processing

Implement the `chat` method to process chat completion requests:

```typescript
async chat(messages: Message[], options: ApiRequestOptions = {}): Promise<ChatCompletionResponse> {
  // Prepare request data
  const model = options.model as string || this.getDefaultModel();
  
  const requestData: YourBackendRequest = {
    model,
    messages,
    max_tokens: options.max_tokens as number,
    temperature: options.temperature as number,
    top_p: options.top_p as number
    // Include other options as needed
  };

  try {
    // Process the request
    const response = await this.makePostRequest(
      this.apiEndpoint,
      requestData,
      { 'Authorization': `Bearer ${this.getApiKey(this.metadata)}` }
    );
    
    const backendResponse = response as YourBackendResponse;
    
    // Convert to standard format
    return {
      id: backendResponse.id,
      model: backendResponse.model,
      content: backendResponse.choices[0]?.message?.content || '',
      text: backendResponse.choices[0]?.message?.content || '',
      usage: backendResponse.usage
    };
  } catch (error) {
    throw error;
  }
}
```

### 4. Streaming Support

Implement the `streamChat` method to support streaming responses:

```typescript
async *streamChat(messages: Message[], options: ApiRequestOptions = {}): AsyncGenerator<StreamChunk> {
  // Prepare request data
  const model = options.model as string || this.getDefaultModel();
  
  const requestData: YourBackendRequest = {
    model,
    messages,
    stream: true,
    max_tokens: options.max_tokens as number,
    temperature: options.temperature as number,
    top_p: options.top_p as number
    // Include other options as needed
  };
  
  try {
    // Set up streaming request
    const response = await fetch(this.apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getApiKey(this.metadata)}`
      },
      body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || `API error (${response.status})`);
    }
    
    if (!response.body) {
      throw new Error('Response body is null');
    }
    
    // Process the stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      // Handle decoding and yielding chunks
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.trim() === '') continue;
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content || '';
            fullText += content;
            
            yield {
              id: parsed.id || '',
              text: fullText,
              delta: content,
              done: false
            };
          } catch (e) {
            console.warn('Failed to parse stream data:', data);
          }
        }
      }
    }
    
    // Yield final chunk with done flag
    yield {
      id: '',
      text: fullText,
      delta: '',
      done: true
    };
  } catch (error) {
    throw error;
  }
}
```

### 5. Error Handling

Implement comprehensive error handling:

```typescript
private async makePostRequest(url: string, data: any, headers: Record<string, string> = {}): Promise<any> {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...headers
      },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      
      // Map API-specific errors to standard errors
      let errorType = 'api_error';
      
      if (response.status === 401) {
        errorType = 'authentication_error';
      } else if (response.status === 429) {
        errorType = 'rate_limit_error';
      } else if (response.status >= 500) {
        errorType = 'server_error';
      }
      
      throw new Error(`API error (${response.status}): ${errorData.error?.message || 'Unknown error'}`);
    }
    
    return await response.json();
  } catch (error) {
    // Add request context to the error
    error.message = `Request to ${url} failed: ${error.message}`;
    throw error;
  }
}
```

## Testing Your Backend

Create a test file for your backend in the `test/api_backends` directory:

```typescript
// test/api_backends/your_backend.test.ts
import { YourBackend } from '../src/api_backends/your_backend_name';
import { ApiMetadata, Message } from '../src/api_backends/types';

// Mock fetch for testing
global.fetch = jest.fn();

describe('YourBackend API Backend', () => {
  let backend: YourBackend;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        id: 'test-id',
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: 'Hello, I am an AI assistant.'
            },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30
        }
      })
    });
    
    // Set up test data
    mockMetadata = {
      your_backend_api_key: 'test-api-key'
    };
    
    // Create backend instance
    backend = new YourBackend({}, mockMetadata);
  });
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
  });
  
  test('should get API key from metadata', () => {
    // @ts-ignore - Testing protected method
    const apiKey = backend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
  });
  
  // Add more tests for your backend functionality
});
```

## Creating an Example File

Create an example file in the `examples` directory:

```typescript
// examples/your_backend_example.ts
import { YourBackend } from '../src/api_backends/your_backend_name';
import { Message } from '../src/api_backends/types';

async function main() {
  console.log('YourBackend API Example');
  
  try {
    // Initialize the backend
    const backend = new YourBackend(
      {}, // Resources (not needed for basic usage)
      {
        your_backend_api_key: process.env.YOUR_BACKEND_API_KEY || 'your_api_key',
        model: 'your-model-name'
      }
    );
    
    console.log('\n1. Backend initialized');
    
    // Basic chat completion
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is machine learning?' }
    ];
    
    const response = await backend.chat(messages, {
      temperature: 0.7,
      max_tokens: 150
    });
    
    console.log('\n2. Chat completion:');
    console.log(response.text);
    
    // Streaming chat completion
    console.log('\n3. Streaming chat completion:');
    
    const streamingMessages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Write a short story about a robot.' }
    ];
    
    const stream = backend.streamChat(streamingMessages, {
      temperature: 0.8,
      max_tokens: 200
    });
    
    for await (const chunk of stream) {
      process.stdout.write(chunk.delta);
    }
    
    console.log('\n\nExample completed successfully');
  } catch (error) {
    console.error('Example encountered an error:', error);
  }
}

// Call the main function
if (require.main === module) {
  main().catch(error => {
    console.error('Error in main:', error);
    process.exit(1);
  });
}
```

## Creating Documentation

Create a documentation file in the `docs/api_backends` directory using the following template:

```markdown
# YourBackend Usage Guide

## Introduction

[Brief description of your backend and its purpose]

## Prerequisites

- [List of prerequisites]

## Installation

[Installation instructions]

## Basic Usage

[Basic usage examples]

## Advanced Usage

[Advanced usage examples]

## API Reference

[Detailed API reference]

## Best Practices

[Best practices for using your backend]

## Additional Resources

[Links to additional resources]
```

## Adding to README

Add your backend to the README.md file:

1. Add it to the available backends table
2. Add it to the TypeScript migration status list
3. Update the overall migration progress percentage

## Final Checklist

Ensure you have:

- [ ] Created the backend implementation
- [ ] Created type definitions
- [ ] Implemented error handling
- [ ] Added tests
- [ ] Created an example file
- [ ] Added documentation
- [ ] Updated the README

## Tips for Success

1. **Error Handling**: Implement comprehensive error handling to provide clear error messages to users.

2. **Consistent API**: Follow the standard API format used by other backends for consistency.

3. **Authentication**: Support both direct API key passing and environment variables.

4. **Documentation**: Document all parameters, options, and behaviors to make your backend easy to use.

5. **Testing**: Create thorough tests that cover error cases and edge conditions.

6. **Examples**: Provide well-commented examples that demonstrate all key features.

7. **Circuit Breaker**: Implement circuit breaker pattern to prevent cascading failures.

8. **Resource Management**: Consider implementing resource pooling for efficient backend usage.

## Need Help?

For questions about implementing API backends, please consult the existing backend implementations or open an issue in the repository.