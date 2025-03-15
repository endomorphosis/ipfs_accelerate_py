// Tests for OpenAI API Backend
import { OpenAI } from '../../src/api_backends/openai';
import { ApiMetadata, ApiRequestOptions, Message } from '../../src/api_backends/types';

// Mock global fetch
global.fetch = jest.fn();
const mockFetch = global.fetch as jest.Mock;

describe('OpenAI API Backend', () => {
  let backend: OpenAI;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Clear mocks before each test
    jest.clearAllMocks();
    
    // Set up mock metadata with API key
    mockMetadata = {
      openai_api_key: 'test-api-key'
    };
    
    // Create instance with mock metadata
    backend = new OpenAI({}, mockMetadata);
    
    // Reset fetch mock
    mockFetch.mockReset();
    
    // Default mock response for simple tests
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'chatcmpl-123',
        object: 'chat.completion',
        created: 1678888888,
        model: 'gpt-4o',
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: 'Hello, I am an AI assistant.'
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: 20,
          completion_tokens: 10,
          total_tokens: 30
        }
      })
    });
  });
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
  });
  
  test('should get API key from metadata', () => {
    // @ts-ignore - Testing protected method
    const apiKey = backend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
  });
  
  test('should get default model', () => {
    // @ts-ignore - Testing protected method
    const model = backend.getDefaultModel();
    expect(model).toBe('gpt-4o');
  });
  
  test('should check model compatibility', () => {
    expect(backend.isCompatibleModel('gpt-4')).toBe(true);
    expect(backend.isCompatibleModel('text-embedding-3-small')).toBe(true);
    expect(backend.isCompatibleModel('text-moderation-latest')).toBe(true);
    expect(backend.isCompatibleModel('dall-e-3')).toBe(true);
    expect(backend.isCompatibleModel('tts-1')).toBe(true);
    expect(backend.isCompatibleModel('whisper-1')).toBe(true);
    expect(backend.isCompatibleModel('llama-7b')).toBe(false);
  });
  
  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
  });
  
  test('should test endpoint', async () => {
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(mockFetch).toHaveBeenCalled();
    expect(mockFetch.mock.calls[0][0]).toContain('/embeddings');
  });
  
  test('should handle chat completion', async () => {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.id).toBe('chatcmpl-123');
    expect(response.model).toBe('gpt-4o');
    expect(response.content).toBe('Hello, I am an AI assistant.');
    
    // Verify fetch was called with correct parameters
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        },
        body: expect.any(String)
      })
    );
    
    // Verify the request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'gpt-4o',
      messages: messages
    });
  });
  
  test('should handle function calling', async () => {
    // Mock successful response with function calling
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'chatcmpl-456',
        object: 'chat.completion',
        created: 1678888889,
        model: 'gpt-4o',
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [{
              id: 'call_abc123',
              type: 'function',
              function: {
                name: 'get_weather',
                arguments: '{"location":"San Francisco","unit":"celsius"}'
              }
            }]
          },
          finish_reason: 'tool_calls'
        }],
        usage: {
          prompt_tokens: 25,
          completion_tokens: 15,
          total_tokens: 40
        }
      })
    });

    // Test messages
    const messages: Message[] = [
      { role: 'user', content: 'What is the weather in San Francisco?' }
    ];

    // Function definitions for testing
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

    // Call chat method with functions
    const result = await backend.chat(messages, { tools, toolChoice: 'auto' });

    // Verify fetch was called with correct parameters
    expect(mockFetch).toHaveBeenCalledTimes(1);

    // Verify the request body includes tools
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'gpt-4o',
      messages: messages,
      tools: tools,
      tool_choice: 'auto'
    });

    // Verify response includes tool calls
    expect(result.tool_calls).toBeDefined();
    expect(result.tool_calls[0].function.name).toBe('get_weather');
  });
  
  test('should handle embeddings', async () => {
    // Mock successful response for embeddings
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        object: 'list',
        data: [{
          object: 'embedding',
          embedding: [0.1, 0.2, 0.3, 0.4],
          index: 0
        }],
        model: 'text-embedding-3-small',
        usage: {
          prompt_tokens: 8,
          total_tokens: 8
        }
      })
    });

    // Call embedding method
    const result = await backend.embedding('Hello world');

    // Verify fetch was called with correct parameters
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/embeddings',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        },
        body: expect.any(String)
      })
    );

    // Verify the request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'text-embedding-3-small',
      input: 'Hello world',
      encoding_format: 'float'
    });

    // Verify response format
    expect(result).toEqual([[0.1, 0.2, 0.3, 0.4]]);
  });
  
  test('should handle moderation', async () => {
    // Mock successful response for moderation
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'modr-123',
        model: 'text-moderation-latest',
        results: [{
          flagged: true,
          categories: {
            hate: false,
            'hate/threatening': false,
            'self-harm': false,
            sexual: false,
            'sexual/minors': false,
            violence: true,
            'violence/graphic': false
          },
          category_scores: {
            hate: 0.01,
            'hate/threatening': 0.01,
            'self-harm': 0.01,
            sexual: 0.01,
            'sexual/minors': 0.01,
            violence: 0.85,
            'violence/graphic': 0.01
          }
        }]
      })
    });

    // Call moderation method
    const result = await backend.moderation('I want to harm someone');

    // Verify fetch was called with correct parameters
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/moderations',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(String)
      })
    );

    // Verify the request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      input: 'I want to harm someone',
      model: 'text-moderation-latest'
    });

    // Verify response
    expect(result.results[0].flagged).toBe(true);
    expect(result.results[0].categories.violence).toBe(true);
  });
  
  test('should handle image generation', async () => {
    // Mock successful response for image generation
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        created: 1684401833,
        data: [{
          revised_prompt: 'A photorealistic image of a kitten',
          url: 'https://example.com/image.png'
        }]
      })
    });

    // Call textToImage method
    const result = await backend.textToImage('A cute kitten');

    // Verify fetch was called with correct parameters
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/images/generations',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(String)
      })
    );

    // Verify the request body
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody).toEqual({
      model: 'dall-e-3',
      prompt: 'A cute kitten',
      n: 1,
      size: '1024x1024',
      style: 'vivid',
      quality: 'standard',
      response_format: 'url'
    });

    // Verify response
    expect(result.data[0].url).toBe('https://example.com/image.png');
  });
  
  test('should handle API errors', async () => {
    // Mock error response
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({
        error: {
          message: 'Invalid API key',
          type: 'authentication_error'
        }
      })
    });
    
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    // Expect the chat method to throw an error
    await expect(backend.chat(messages)).rejects.toMatchObject({
      message: 'Invalid API key',
      statusCode: 401,
      type: 'authentication_error',
      isAuthError: true
    });
  });
  
  test('should handle rate limiting', async () => {
    // Mock rate limit response
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 429,
      headers: {
        get: (name: string) => name === 'retry-after' ? '2' : null
      },
      json: async () => ({
        error: {
          message: 'Rate limit exceeded',
          type: 'rate_limit_error'
        }
      })
    });

    // Mock successful response after retry
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'chatcmpl-123',
        object: 'chat.completion',
        created: 1678888888,
        model: 'gpt-4o',
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: 'This is a retry response'
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: 20,
          completion_tokens: 10,
          total_tokens: 30
        }
      })
    });

    // Override retryableRequest to avoid actually waiting
    jest.spyOn(backend as any, 'retryableRequest').mockImplementationOnce(async (fn: Function) => {
      try {
        return await fn();
      } catch (error) {
        if (error.isRateLimitError) {
          return await fn(); // Immediate retry for testing
        }
        throw error;
      }
    });

    // Call chat method
    const result = await backend.chat([{ role: 'user', content: 'Hello' }]);

    // Verify fetch was called twice
    expect(mockFetch).toHaveBeenCalledTimes(2);

    // Verify result is from second call
    expect(result.content).toBe('This is a retry response');
  });
  
  test('should support streaming', async () => {
    // Create a mock ReadableStream that emits SSE formatted data
    const mockStream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n'));
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'));
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n'));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      }
    });

    // Mock the fetch to return our stream
    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: mockStream
    });

    // Call streamChat method
    const messages: Message[] = [{ role: 'user', content: 'Hi' }];
    const generator = backend.streamChat(messages);

    // Collect stream chunks
    const chunks = [];
    for await (const chunk of generator) {
      chunks.push(chunk);
    }

    // Verify fetch was called with stream: true
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(requestBody.stream).toBe(true);

    // Verify chunks were processed correctly
    expect(chunks.length).toBe(3);
    expect(chunks[0].role).toBe('assistant');
    expect(chunks[1].content).toBe('Hello');
    expect(chunks[2].content).toBe(' world');
  });
});
