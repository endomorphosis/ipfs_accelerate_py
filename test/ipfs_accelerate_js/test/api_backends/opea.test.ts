// Tests for OPEA API Backend
import { OPEA } from '../../src/api_backends/opea';
import { ApiMetadata, Message, ApiRequestOptions } from '../../src/api_backends/types';
import { OPEARequestData, OPEAResponse } from '../../src/api_backends/opea/types';

// Mock fetch for testing
global.fetch = jest.fn();

// Mock ReadableStream for testing streaming
const mockReadableStream = () => {
  const chunks = [
    `data: ${JSON.stringify({
      choices: [{ delta: { content: 'Hello', role: 'assistant' }, finish_reason: null }]
    })}`,
    `data: ${JSON.stringify({
      choices: [{ delta: { content: ', world!', role: 'assistant' }, finish_reason: null }]
    })}`,
    `data: ${JSON.stringify({
      choices: [{ delta: { content: '', role: 'assistant' }, finish_reason: 'stop' }]
    })}`,
    'data: [DONE]'
  ];
  
  let currentIndex = 0;
  
  return {
    getReader: () => ({
      read: () => {
        if (currentIndex < chunks.length) {
          const chunk = chunks[currentIndex];
          currentIndex++;
          return Promise.resolve({
            done: false,
            value: new TextEncoder().encode(chunk + '\n')
          });
        } else {
          return Promise.resolve({ done: true, value: undefined });
        }
      },
      cancel: jest.fn(),
      releaseLock: jest.fn()
    })
  };
};

describe('OPEA API Backend', () => {
  let backend: OPEA;
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
      opea_api_key: 'test-api-key'
    };
    
    // Create backend instance
    backend = new OPEA({}, mockMetadata);
  });
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    
    // Test with custom API URL
    const customBackend = new OPEA({}, { 
      opea_api_url: 'https://custom-opea-url.com',
      opea_model: 'custom-model'
    });
    expect(customBackend).toBeDefined();
    
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.apiUrl).toBe('https://custom-opea-url.com');
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.defaultModel).toBe('custom-model');
  });
  
  test('should get API key from metadata', () => {
    // @ts-ignore - Testing protected method
    const apiKey = backend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
    
    // Test environment variable fallback
    const originalEnv = process.env.OPEA_API_KEY;
    process.env.OPEA_API_KEY = 'env-api-key';
    // @ts-ignore - Testing protected method
    const envApiKey = backend.getApiKey({});
    expect(envApiKey).toBe('env-api-key');
    
    // Restore original environment
    process.env.OPEA_API_KEY = originalEnv;
  });
  
  test('should get default model', () => {
    // @ts-ignore - Testing protected method
    const model = backend.getDefaultModel();
    expect(model).toBeDefined();
    expect(typeof model).toBe('string');
    expect(model).toBe('gpt-3.5-turbo');
  });
  
  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
    
    // Test with custom URL
    const customHandler = backend.createEndpointHandler('https://custom-endpoint.com');
    expect(customHandler).toBeDefined();
    expect(typeof customHandler).toBe('function');
  });
  
  test('should test endpoint', async () => {
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with custom URL
    const customResult = await backend.testEndpoint('https://custom-endpoint.com');
    expect(customResult).toBe(true);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should make POST request to OPEA', async () => {
    // @ts-ignore - Testing protected method
    const response = await backend.makePostRequestOPEA(
      'https://opea-api.com/v1/chat/completions',
      { messages: [{ role: 'user', content: 'Hello' }], max_tokens: 10 }
    );
    
    expect(response).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      'https://opea-api.com/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: expect.any(String)
      })
    );
    
    // Test with options
    // @ts-ignore - Testing protected method
    const responseWithOptions = await backend.makePostRequestOPEA(
      'https://opea-api.com/v1/chat/completions',
      { messages: [{ role: 'user', content: 'Hello' }], max_tokens: 10 },
      { temperature: 0.5, top_p: 0.9, apiKey: 'custom-key' }
    );
    
    expect(responseWithOptions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch).toHaveBeenLastCalledWith(
      'https://opea-api.com/v1/chat/completions',
      expect.objectContaining({
        body: expect.stringContaining('"temperature":0.5')
      })
    );
  });
  
  test('should make streaming request to OPEA', async () => {
    // Mock streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    // @ts-ignore - Testing protected method
    const stream = await backend.makeStreamRequestOPEA(
      'https://opea-api.com/v1/chat/completions',
      { messages: [{ role: 'user', content: 'Hello' }], max_tokens: 10 }
    );
    
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    expect(global.fetch).toHaveBeenCalledWith(
      'https://opea-api.com/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
  });
  
  test('should handle chat completion', async () => {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.content).toBe('Hello, I am an AI assistant.');
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with options
    const responseWithOptions = await backend.chat(messages, {
      model: 'custom-model',
      temperature: 0.5,
      top_p: 0.9,
      max_tokens: 50
    });
    
    expect(responseWithOptions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should stream chat completion', async () => {
    // Mock streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const stream = backend.streamChat(messages);
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of await stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/chat/completions'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
  });
  
  test('should implement makePostRequest abstract method', async () => {
    const data = { 
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 10
    };
    const response = await backend.makePostRequest(data);
    
    expect(response).toBeDefined();
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with custom endpoint
    const customResponse = await backend.makePostRequest(data, undefined, {
      endpoint: 'https://custom-endpoint.com'
    });
    
    expect(customResponse).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch).toHaveBeenLastCalledWith(
      'https://custom-endpoint.com',
      expect.any(Object)
    );
  });
  
  test('should implement makeStreamRequest abstract method', async () => {
    // Need to reset the mock counter for this test
    jest.clearAllMocks();
    
    // Mock streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    const data = { 
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 10
    };
    const stream = await backend.makeStreamRequest(data);
    
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });
  
  test('should check model compatibility', () => {
    // Test with compatible model prefixes
    expect(backend.isCompatibleModel('gpt-3.5-turbo')).toBe(true);
    expect(backend.isCompatibleModel('text-davinci-003')).toBe(true);
    expect(backend.isCompatibleModel('claude-v1')).toBe(true);
    expect(backend.isCompatibleModel('llama-7b')).toBe(true);
    
    // Test with incompatible model
    expect(backend.isCompatibleModel('')).toBe(false);
  });
});
